import os
import logging
import asyncio
from typing import List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from agents.novel2script.scene_extractor import Scene
from langchain.output_parsers import PydanticOutputParser
from agents.elements.event import Event

system_prompt_template_merge_characters_across_scenes = \
"""
You are an expert script analysis and character fusion specialist. Your role is to intelligently analyze multiple script scenes, identify characters that represent the same entity across different scenes, and merge them into a unified character list with consistent identifiers.

**TASK**
Process the input scenes, each containing a script and character names, and output a consolidated list of characters. Each character in the list must have a unique identifier, along with the scene numbers where they appear and the name used in each scene.

**INPUT**
A sequence of scenes. Each scene is enclosed within <SCENE_N_START> and <SCENE_N_END> tags, where N is the scene number(starting from 0). Each scene includes a screnplay script and a sequence of character names. The screenplay script is enclosed within <SCRIPT_START> and <SCRIPT_END> tags. The sequence of character names is enclosed within <Characters_START> and <Characters_END> tags. Each character in the list is enclosed within <CHARACTER_M_START> and <CHARACTER_M_END> tags, where M is the character number(starting from 0).

Below is an example of one scene:

<SCENE_0_START>

<SCRIPT_START>
John enters the room and sees Mary.
JOHN: Hi Mary, how are you?
Mary: I'm good, John. Thanks for asking!
<SCRIPT_END>

<Characters_START>
<CHARACTER_0_START>John<CHARACTER_0_END>
<CHARACTER_1_START>Mary<CHARACTER_1_END>
<Characters_END>

<SCENE_0_END>



**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Character Fusion: Analyze contextual clues (e.g., dialogue style, role in plot, relationships, descriptions) to determine if characters from different scenes are the same person, even if names vary.
2. Unique Identifier: Assign a consistent, unique ID (e.g., primary/canonical name) to each merged character. Use the most frequent or contextually appropriate name as the identifier, if possible.
3. Scene Mapping: For each character, list all scenes they appear in and the exact name used in each scene.
4. Completeness: Ensure all characters from all scenes are included in the final list. No duplicate, omitted, or extraneous characters.
5. If a character undergoes significant changes across different scenes, it is necessary to split them into separate roles. For example, if Character A is a child in Scene 0 but an adult in Scene 1, they should be divided into two distinct characters (meaning two different actors are required to portray them).
"""


human_prompt_template_merge_characters_across_scenes = \
"""
{scenes_sequence}
"""



system_prompt_template_merge_characters_across_events = \
"""
You are an expert in role fusion and consistency management across multiple narrative scripts. Your expertise includes analyzing character attributes, dialogue patterns, contextual behaviors, and narrative functions to accurately identify and merge references to the same underlying character, even when they use different names or appear in distinct events written by different authors.

**TASK**
Analyze all provided events (each with a description and a list of roles) to fuse character references that logically represent the same entity. For each fused character, generate a unique identifier and list all occurrences—specifying the event number and the local name used in that event. Output a consolidated list of such fused roles.

**INPUT**
A sequence of events. Each event is enclosed within <EVENT_N_START> and <EVENT_N_END> tags, where N is the event number(starting from 0). Each event includes a detailed description and a sequence of character names. The description is enclosed within <DESCRIPTION_START> and <DESCRIPTION_END> tags. The sequence of character names is enclosed within <Characters_START> and <Characters_END> tags. Each character in the list is enclosed within <CHARACTER_M_START> and <CHARACTER_M_END> tags, where M is the character number(starting from 0).


Below is an example of one event:
<EVENT_0_START>
<DESCRIPTION_START>
On a sunny Saturday afternoon at around 3 PM, in a small, cozy city park, a young woman named Lily decided to read her new book on a bench under a large oak tree. The cause was simply her desire to enjoy some quiet time outdoors after a busy week. As she was reading, a sudden gust of wind blew the pages wildly, causing her to lose her place, and then the wind snatched the bookmark—a dried flower from her grandmother's garden—and sent it fluttering across the grass. She quickly got up and chased after it, slightly panicked as it was very precious to her. The bookmark led her on a brief dance across the lawn before it landed near the foot of an elderly man who was feeding pigeons. Noticing her distress, he gently picked it up and handed it to her with a warm smile. Grateful and relieved, Lily thanked him, and they ended up sharing the bench, talking about books and memories for over an hour. The result was not only the recovery of her cherished bookmark but also the beginning of a delightful new friendship.
<DESCRIPTION_END>

<Characters_START>
<CHARACTER_0_START>Lily<CHARACTER_0_END>
<CHARACTER_1_START>An Elderly Man<CHARACTER_1_END>
<Characters_END>

<EVENT_0_END>

**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Character Fusion: Merge characters only if they share strong consistent traits—such as narrative function, relationships, key attributes, or behavioral patterns—across events. Avoid over-fusion; distinct characters should remain separate.
2. Unique Identifier: Assign a consistent, unique ID (e.g., primary/canonical name) to each merged character. Use the most frequent or contextually appropriate name as the identifier, if possible.
3. Scene Mapping: For each character, list all events they appear in and the exact name used in each event.
4. Completeness: Ensure all characters from all events are included in the final list. No duplicate, omitted, or extraneous characters.
5. If a character undergoes significant changes across different events, it is necessary to split them into separate roles. For example, if Character A is a child in Event 0 but an adult in Event 1, they should be divided into two distinct characters (meaning two different actors are required to portray them).
"""

human_prompt_template_merge_characters_across_events = \
"""
{events_sequence}
"""


class CharacterAcrossScene(BaseModel):
    identifier: str = Field(
        ...,
        description="Unique identifier for the character",
        examples=["Alice Smith", "Bob Johnson"],
    )
    identifier_in_specific_scenes: List[Tuple[int, str]] = Field(
        description="List of scene identifiers and their corresponding character names",
        examples=[[(0, "Alice"), (2, "Alice Smith")], [(1, "Bob"), (3, "Bob Johnson")]],
    )



class CharactersAcrossSceneResponse(BaseModel):
    characters: List[CharacterAcrossScene] = Field(
        description="List of merged characters with their identifiers and scene mappings",
    )


class CharacterAcrossEvent(BaseModel):
    identifier: str = Field(
        ...,
        description="Unique identifier for the character",
        examples=["Alice Smith", "Bob Johnson"],
    )
    identifier_in_specific_events: List[Tuple[int, str]] = Field(
        description="List of event identifiers and their corresponding character names",
        examples=[[(0, "Alice"), (2, "Alice Smith")], [(1, "Bob"), (3, "Bob Johnson")]],
    )


class CharactersAcrossEventResponse(BaseModel):
    characters: List[CharacterAcrossEvent] = Field(
        description="List of merged characters with their identifiers and event mappings",
    )


class CharacterMerger:
    def __init__(
        self,
        chat_model: str,
        api_key: str,
        base_url: str,
    ):
        self.chat_model = init_chat_model(
            model=chat_model,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
        )


    def merge_character_across_scenes(
        self,
        scenes: List[Scene]
    ):
        scenes_sequence_str = ""
        for scene in scenes:
            scene_str = f"<SCENE_{scene.idx}_START>\n\n"
            scene_str += "<SCRIPT_START>\n"
            scene_str += scene.script + "\n"
            scene_str += "<SCRIPT_END>\n\n"
            scene_str += "<Characters_START>\n"
            for i, character in enumerate(scene.characters):
                scene_str += f"<CHARACTER_{i}_START>{character}<CHARACTER_{i}_END>\n"
            scene_str += "<Characters_END>\n\n"
            scene_str += f"<SCENE_{scene.idx}_END>\n\n"
            scenes_sequence_str += scene_str

        parser = PydanticOutputParser(pydantic_object=CharactersAcrossSceneResponse)

        messages = [
            SystemMessage(
                content=system_prompt_template_merge_characters_across_scenes.format(
                    format_instructions=parser.get_format_instructions(),
                ),
            ),
            HumanMessage(
                content=human_prompt_template_merge_characters_across_scenes.format(
                    scenes_sequence=scenes_sequence_str,
                )
            )
        ]

        chain = self.chat_model | parser
        response: CharactersAcrossSceneResponse = chain.invoke(messages)
        return response.characters




    def merge_character_across_events(
        self,
        events: List[Event],
        characters_each_events: List[List[CharacterAcrossScene]],
    ):
        events_sequence_str = ""
        for event, characters in zip(events, characters_each_events):
            event_str = f"<EVENT_{event.idx}_START>\n\n"
            event_str += "<DESCRIPTION_START>\n"
            event_str += event.description + "\n"
            event_str += "<DESCRIPTION_END>\n\n"
            event_str += "<Characters_START>\n"
            for i, character in enumerate(characters):
                event_str += f"<CHARACTER_{i}_START>{character.identifier}<CHARACTER_{i}_END>\n"
            event_str += "<Characters_END>\n\n"
            event_str += f"<EVENT_{event.idx}_END>\n\n"
            events_sequence_str += event_str

        parser = PydanticOutputParser(pydantic_object=CharactersAcrossEventResponse)

        messages = [
            SystemMessage(
                content=system_prompt_template_merge_characters_across_events.format(
                    format_instructions=parser.get_format_instructions(),
                ),
            ),
            HumanMessage(
                content=human_prompt_template_merge_characters_across_events.format(
                    events_sequence=events_sequence_str,
                )
            )
        ]

        chain = self.chat_model | parser
        response: CharactersAcrossEventResponse = chain.invoke(messages)
        return response.characters


