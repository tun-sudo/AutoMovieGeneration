from openai import OpenAI
import base64
from typing import Optional, Union, List
import requests
from agents.storyboard_generator import Character, Shot
import os
import logging
from agents.elements import Character, Shot
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from agents.elements import Character, Shot
import json

system_prompt_template_select_reference_images = \
"""
You are a professional visual creation assistant skilled in multimodal image analysis and reasoning.

**Task**
Your core task is to intelligently select the most suitable reference images from a provided reference image library (including multiple character reference images and existing scene images from prior shots) based on the user's text description (describing the target shot), ensuring that the subsequently generated image meets the following key consistencies:
(1) Character Consistency: The appearance (e.g., facial features, hairstyle, body shape), clothing, expression, posture, etc., of the generated character should highly match the reference images.
(2) Environmental Consistency: The scene of the generated image (e.g., background, lighting, atmosphere, layout) should remain coherent with the existing images from prior shots.
(3) Style Consistency: The visual style of the generated image (e.g., realistic, cartoon, film-like, color tone) should harmonize with the reference images and existing images.


**Input**
You will receive a text description of the target shot, along with a set of reference images. The text description of the target shot is enclosed within <SHOT_DESCRIPTION_START> and <SHOT_DESCRIPTION_END>. Each reference image is provided with a brief text description. The reference images are indexed starting from 0.


**Output**
You need to select the most relevant reference images based on the user's description and put the corresponding indices in the `ref_image_indices` field of the output. At the same time, you should generate a text prompt that describes the image to be created, specifying which elements in the generated image should reference which image (and which elements within it).

{format_instructions}


**Note**
1. The reference images may depict the same character from different angles, in different outfits, or in different scenes. Identify the image closest to the version described by the user.
2. Ensure maximum visual consistency across character appearance, environmental context, and stylistic elements. Enable coherent visual storytelling across sequential images
3. Prioritize images with similar compositions (shot by the same camera).
4. Choose reference images that are as concise as possible and avoid including duplicate information. Not more than 5 images should be selected.
5. Ensure that the language of all output values(not include keys) matches that used in the shot description.
"""


human_prompt_template_select_reference_images = \
"""
<SHOT_DESCRIPTION_START>
{shot_description}
<SHOT_DESCRIPTION_END>
"""


def encode_base64(file_path):
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"



class RefImageIndicesAndTextPrompt(BaseModel):
    ref_image_indices: List[int] = Field(
        ...,
        description="Indices of reference images selected from the provided images. For example, [0, 2, 5] means selecting the first, third, and sixth images. The indices should be 0-based."
    )
    text_prompt: str = Field(
        ...,
        description="Text description to guide the image generation. You need to describe the image to be generated, specifying which elements in the generated image should reference which image (and which elements within it). For example, 'Create an image following the given description: \nThe man is standing in the landscape. The man should reference the Image 0. The landscape should reference the Image 1.' Here, the index of the reference image should refer to its position in the ref_image_indices list, not the sequence number in the provided image list."
    )



class ReferenceImageSelector:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_provider: str = "openai",
        model: str = "claude-sonnet-4-20250514-thinking",
    ):
        self.base_url = base_url
        self.api_key = api_key

        self.chat_model = init_chat_model(
            model=model,
            model_provider=model_provider,
            base_url=base_url,
            api_key=api_key,
        )


    @retry
    def __call__(
        self,
        available_image_path_and_text_pairs: List[Tuple[str, str]],
        frame_description: str,
    ):
        human_content = []
        for idx, (image_url, text) in enumerate(available_image_path_and_text_pairs):
            human_content.append({
                "type": "text",
                "text": f"Image {idx}: {text}"
            })
            human_content.append({
                "type": "image_url",
                "image_url": {"url": encode_base64(image_url)}
            })
        human_content.append({
            "type": "text",
            "text": human_prompt_template_select_reference_images.format(shot_description=frame_description)
        })

        parser = PydanticOutputParser(pydantic_object=RefImageIndicesAndTextPrompt)

        messages = [
            SystemMessage(content=system_prompt_template_select_reference_images.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_content)
        ]

        chain = self.chat_model | parser

        try:
            response = chain.invoke(messages)        

        except Exception as e:
            logging.error(f"Error get image prompt: \n{e}")
            raise e

        return response

