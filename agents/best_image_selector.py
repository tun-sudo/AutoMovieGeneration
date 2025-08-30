import logging
from typing import List, Tuple
from pydantic import BaseModel, Field
from tenacity import retry
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from agents.utils.image import image_to_base64_with_mime



system_prompt_template_select_most_consistent_image = \
"""
You are a professional visual assessment expert. Your expertise includes identifying Character Consistency and Spatial Consistency between candidate image and reference image, and assessing semantic consistency between candidate image and text description.

**TASK**
Based on the reference image provided by the user, the text description of the target image, and several candidate images, evaluate which candidate image performs best in the following aspects:
1.Character Consistency: Whether the character features (a. gender, b.ethnicity, c.age, d.facial features, e.body shape, f.outlook, g. hairstyle) in the candidate image align with those of the character in the reference image.
2.Spatial Consistency: Whether the relative positions between characters (e.g. Character A is on the left, character B is on the right, scene layout, perspective, and other spatial relationships) in the candidate image are consistent with those in the reference image.
3.Description Accuracy: Whether the candidate image accurately reflects the content described in the text (Note: The text description describes the target image we want, which is not an editing instruction).

**INPUT**
The user will provide the following content:
- Reference images: These include images of characters or other perspectives, each along with a brief text description. For example, "Reference Image 0: A young girl with long brown hair wearing a red dress." then follow the corresponding image. The index starts from 0.
- Candidate images: The candidate images to be evaluated. For example, "Generated Image 0", then follow a generated image. The index starts from 0.
- Text description for target image: This describes what the generated image should contain. It is enclosed <TARGET_DESCRIPTION_START> and <TARGET_DESCRIPTION_END> tags.

**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Prioritize Character Consistency: Ensure that the characters in the generated image are highly consistent with those in the reference image in terms of visual features (e.g., a. gender b.ethnicity, c.age, d.facial features, e.body shape, f.outlook, g. hairstyle etc.).
2. Focus on Spatial Consistency: Verify whether the relative positions of characters, object arrangements, and perspectives align logically with the reference image (e.g., if Character A is on the left and Character B is on the right in the reference image, the generated image should not reverse this).
3. Strictly Compare with Text Description: The generated image must adhere to key elements in the text description (e.g., actions, scenes, objects, etc.), while disregarding parts related to editing instructions (as the input description reflects the expected outcome rather than directives).
4. If multiple images partially meet the criteria, select the one with the highest overall consistency; if none are ideal, choose the relatively best option and explain its shortcomings.
5. Avoid subjective preferences; base all analysis on objective comparisons.
"""

human_prompt_template_select_most_consistent_image = \
"""
<TARGET_DESCRIPTION_START>
{target_description}
<TARGET_DESCRIPTION_END>
"""


class BestImageResponse(BaseModel):
    best_image_index: int = Field(
        ...,
        description="The index of the best image."
    )
    reason: str = Field(
        ...,
        description="The reason why the image is the best."
    )


class BestImageSelector:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
    ):
        
        self.chat_model = init_chat_model(
            model=chat_model,
            model_provider="openai",
            base_url=base_url,
            api_key=api_key,
        )


    @retry
    def __call__(
        self,
        ref_image_path_and_text_pairs: List[Tuple[str, str]],
        target_description: str,
        candidate_image_paths: List[str],
    ) -> str:
        """
        Args:
            ref_image_path_and_text_pairs:
            A list of tuples containing reference image paths and their descriptions.

            target_description:
            The description of the target image.

            candidate_image_paths:
            A list of paths to the candidate images to be evaluated.
        """

        try:
            ref_image_paths = [ref_image_path for ref_image_path, _ in ref_image_path_and_text_pairs]
            logging.info(f"Selecting the best image from candidates: {candidate_image_paths}")

            human_content = []
            for idx, (ref_image_path, text) in enumerate(ref_image_path_and_text_pairs):
                human_content.append({
                    "type": "text",
                    "text": f"Reference Image {idx}: {text}"
                })
                human_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_with_mime(ref_image_path)}
                })

            for idx, candidate_image_path in enumerate(candidate_image_paths):
                human_content.append({
                    "type": "text",
                    "text": f"Candidate Image {idx}"
                })
                human_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_with_mime(candidate_image_path)}
                })
            human_content.append({
                "type": "text",
                "text": human_prompt_template_select_most_consistent_image.format(target_description=target_description)
            })

            parser = PydanticOutputParser(pydantic_object=BestImageResponse)

            messages = [
                SystemMessage(content=system_prompt_template_select_most_consistent_image.format(format_instructions=parser.get_format_instructions())),
                HumanMessage(content=human_content)
            ]

            chain = self.chat_model | parser

            response = chain.invoke(messages)
            best_image_path = candidate_image_paths[response.best_image_index]
            logging.info(f"Best image selected: {best_image_path}")
            logging.info(f"Selection reason: {response.reason}")
            return best_image_path

        except Exception as e:
            logging.error(f"Error selecting the best image: {e}")
            raise e
