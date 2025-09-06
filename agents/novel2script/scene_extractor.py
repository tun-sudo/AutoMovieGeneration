from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from agents.elements import Event
from agents.tools.silicon_rerank import SiliconReranker
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple, Dict
from langchain_core.output_parsers import PydanticOutputParser



system_prompt_template_get_next_scene = \
"""
You are an expert scriptwriter specializing in adapting literary works into structured screenplay scenes. Your task is to analyze event descriptions from novels and transform them into compelling screenplay scenes, leveraging relevant context while ignoring extraneous information.

**TASK**
Generate the next scene for a screenplay adaptation based on the provided input. Each scene must include:
- Setting: Clear description of the environment/time
- Characters: List of characters appearing in the scene
- Script: Character actions and dialogues in standard screenplay format

**INPUT**
- Event Description: A clear, concise summary of the event to adapt. The event description is enclosed within <EVENT_DESCRIPTION_START> and <EVENT_DESCRIPTION_END> tags.
- Context Fragments: Multiple excerpts retrieved from the novel via RAG. These may contain irrelevant passages. Ignore any content not directly related to the event. The sequence of context fragments is enclosed within <CONTEXT_FRAGMENTS_START> and <CONTEXT_FRAGMENTS_END> tags. Each fragment in the sequence is enclosed within its own <FRAGMENT_N_START> and <FRAGMENT_N_END> tags, with N being the fragment number.
- Previous Scenes (if any): Already adapted scenes for context (may be empty). The sequence of previous scenes is enclosed within <PREVIOUS_SCENES_START> and <PREVIOUS_SCENES_END> tags. Each scene is enclosed within its own <SCENE_N_START> and <SCENE_N_END> tags, with N being the scene number.

**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Focus on Relevance: Use only context fragments that directly align with the event description. Disregard any unrelated paragraphs.
2. Dialogues and Actions: Convert descriptive prose into actionable lines and dialogues. Invent minimal necessary dialogue if implied but not explicit in the context.  
3. Conciseness: Keep descriptions brief and visual. Avoid prose-like explanations.  
4. Format Consistency: Ensure industry-standard screenplay structure.
5. Implicit Inference: If context fragments lack exact details (e.g., location/time), infer logically from the event description or broader narrative context.  
6. No Extraneous Content: Do not include scenes, characters, or dialogues unrelated to the core event.
7. Use as few scenes as possible to narrate the events, and only add a new scene when the location or time changes.
8. The language of outputs in values should be same as the input.
9. The total number of scenes should NOT EXCEED 5 !!!!
"""


human_prompt_template_get_next_scene = \
"""
<EVENT_DESCRIPTION_START>
{event_description}
<EVENT_DESCRIPTION_END>

<CONTEXT_FRAGMENTS_START>
{context_fragments}
<CONTEXT_FRAGMENTS_END>

<PREVIOUS_SCENES_START>
{previous_scenes}
<PREVIOUS_SCENES_END>
"""



class Scene(BaseModel):
    idx: int = Field(
        description="The scene index, starting from 0",
        examples=[0, 1, 2],
    )
    is_last: bool = Field(
        description="Indicates if this is the last scene",
        examples=[False, True],
    )
    setting: str = Field(
        description="The scene setting, including location and time",
        examples=["INT. LIBRARY - NIGHT", "EXT. PARK - DAY"],
    )
    characters: List[str] = Field(
        default=[],
        description="List of characters appearing in the script. Each character must be an individual, not a group. Character names should be consistent with those in the novel.",
        examples=[["Jane", "John"], ["Alice", "Bob"]],
    )
    script: str = Field(
        description="The screenplay script for the scene, including character actions and dialogues. Character names in the script should be enclosed in <>, except for character names within dialogues.",
        examples=[
            "<Jane> paces nervously, clutching a letter. She turns to <John>.\n<Jane>: John, we need to leave tonight.\n<John> shakes his head, stepping toward the window.\n<John>: It's too dangerous.",
            "<Alice> sits quietly, observing the chaos around her. She whispers to <Bob>.\n<Alice>: Bob, do you think they'll find us here?\n<Bob> nods slowly, his expression grim."
        ],
    )

    def __str__(self):
        s = f"Scene {self.idx}:"
        s += f"\nSetting: {self.setting}"
        s += f"\nCharacters: {', '.join(self.characters)}"
        s += f"\nScript: \n{self.script}"
        return s

    def __repr__(self):
        return self.__str__()




class SceneExtractor:
    def __init__(
        self,
        api_key,
        base_url,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        knowledge_base_cache_dir: str = ".cache",

        rerank: bool = True,
        rerank_model: str = "BAAI/bge-reranker-v2-m3",

        chat_model: str = "gpt-5-2025-08-07",
    ):

        # construct_knowledge_base
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        underlying_embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        )
        document_embedding_cache = LocalFileStore(
            root_path=knowledge_base_cache_dir
        )
        self.embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=underlying_embeddings,
            document_embedding_cache=document_embedding_cache,
            namespace=underlying_embeddings.model,
            key_encoder="sha256",
        )

        # rerank
        self.rerank_model = None
        if rerank:
            self.rerank_model = SiliconReranker(
                api_key=api_key,
                base_url=base_url,
                model=rerank_model,
            )


        # chat model
        self.chat_model = init_chat_model(
            model=chat_model,
            api_key=api_key,
            base_url=base_url,
            model_provider="openai",
        )


    async def extract_all_scenes(
        self,
        knowledge_base: FAISS,
        event: Event,
    ):
        # take the relevant chunks as context and divide the event into one or more scenes based on the event description
        scenes = []
        while True:
            scene = await self.get_next_scene(knowledge_base, event, scenes)
            scenes.append(scene)
            if scene.is_last:
                break

        return scenes


    def construct_knowledge_base(
        self,
        novel_text,
    ):
        text_chunks = self.text_splitter.split_text(novel_text)
        knowledge_base: FAISS = FAISS.from_texts(text_chunks, self.embeddings)
        return knowledge_base


    def extract_relevant_chunks(
        self,
        knowledge_base: FAISS,
        event: Event,
    ):
        relevant_chunks = knowledge_base.similarity_search_with_relevance_scores(
            query=event.description,
            k=10,
        )
        relevant_chunks = [(doc.page_content, score) for doc, score in relevant_chunks]

        if self.rerank_model:
            documents = [doc for doc, _ in relevant_chunks]
            relevant_chunks = self.rerank_model(
                documents=documents,
                query=event.description,
                top_n=5,
            )

        return relevant_chunks


    def get_next_scene(
        self,
        knowledge_base: FAISS,
        event: Event,
        previous_scenes: List[Scene]
    ) -> Scene:
        relevant_chunks = self.extract_relevant_chunks(knowledge_base, event)

        context_fragments_str = "\n".join([f"<FRAGMENT_{i}_START>\n{chunk}\n<FRAGMENT_{i}_END>" for i, (chunk, _) in enumerate(relevant_chunks)])

        previous_scenes_str = "\n".join([f"<SCENE_{i}_START>\n{scene}\n<SCENE_{i}_END>" for i, scene in enumerate(previous_scenes)])

        parser = PydanticOutputParser(pydantic_object=Scene)

        messages = [
            SystemMessage(
                content=system_prompt_template_get_next_scene.format(
                    format_instructions=parser.get_format_instructions(),
                ),
            ),
            HumanMessage(
                content=human_prompt_template_get_next_scene.format(
                    event_description=event.description,
                    context_fragments=context_fragments_str,
                    previous_scenes=previous_scenes_str,
                )
            )
        ]

        chain = self.chat_model | parser
        scene = chain.invoke(messages)
        return scene
