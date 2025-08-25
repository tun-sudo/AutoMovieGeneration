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


system_prompt_template = \
"""
You are a professional visual content analysis assistant, specializing in comparing and evaluating image consistency.

**Task**
Your task is to determine whether the generated image matches the reference image. You need to strictly compare the consistency between the reference image and the generated image across the following dimensions:
Character Consistency: Including the character's appearance and clothing.
Spatial Consistency: Including the scene layout and spatial relationships between characters.
Style Consistency: Including artistic style, color tone, and lighting effects.

**Input**
You will receive a set of reference images along with their brief text descriptions, a guide prompt used for generating the image, and the generated image itself. The guide prompt is enclosed within <GUIDE_PROMPT_START> and <GUIDE_PROMPT_END>

**Output**
{format_instructions}

**Guidelines**
1. When checking character consistency, it is necessary to verify all characters in the generated image to determine whether they are the same person as the corresponding reference portraits. Only facial features and body type should be evaluated, while pose, movement, emotion, and clothing are excluded from the assessment.
2. When checking spatial consistency, it is necessary to determine whether the background environment is the same, and whether the relative positions, sizes, and proportions of key objects are correct. Additionally, if the generated image and the reference image are not from the same perspective, issues such as missing objects, distorted spatial relationships, and misaligned object positions often arise during the generation process. These issues should also be examined.
3. When checking style consistency, it is necessary to verify whether the generated image and the reference image belong to the same medium or style (such as oil painting, watercolor, anime, pixel art, etc.), whether the color tones match, and whether the overall lighting atmosphere (e.g., sunny afternoon, gloomy dusk, warm indoor lighting) is consistent.
4. If any inconsistencies are detected, they must be explained in the "reason" field, and the "guide prompt" should be revised accordingly based on the identified discrepancies.
"""

human_prompt_template = \
"""
After editing the reference image according to the guidance prompt below, the generated image is shown in the last picture.
<GUIDE_PROMPT_START>
{guide_prompt}
<GUIDE_PROMPT_END>
"""


def encode_base64(file_path):
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"


class ConsistencyCheckerResult(BaseModel):
    is_consistent: bool = Field(
        ...,
        description="Indicates whether the image is consistent with the reference."
    )
    reason: str = Field(
        ...,
        description="The reason for the consistency check result."
    )
    rectified_guide_prompt: Optional[str] = Field(
        None,
        description="The rectified guide prompt to improve image generation, if the generated image is inconsistent. Otherwise, it will be None."
    )


class ImageConsistencyChecker:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514-thinking",
        model_provider: str = "openai",
        base_url: str = "https://yunwu.ai/v1",
        api_key: str = "sk-bapMAyiji5uA2O91yTv6iBIXmp3e0JsMWByxgzHkzOd9jRIF",  
    ):
        self.chat_model = init_chat_model(
            model=model,
            model_provider=model_provider,
            base_url=base_url,
            api_key=api_key,
        )

    @retry
    def __call__(
        self,
        ref_image_path_and_text_pairs: List[Tuple[str, str]],
        guide_prompt: str,
        generated_image_path: str,
    ):
        human_content = []
        for idx, (image_url, text) in enumerate(ref_image_path_and_text_pairs):
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
            "text": human_prompt_template.format(guide_prompt=guide_prompt)
        })
        human_content.append({
            "type": "image_url",
            "image_url": {"url": encode_base64(generated_image_path)}
        })

        parser = PydanticOutputParser(pydantic_object=ConsistencyCheckerResult)

        messages = [
            SystemMessage(content=system_prompt_template.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_content)
        ]

        chain = self.chat_model | parser

        try:
            response = chain.invoke(messages)

        except Exception as e:
            logging.error(f"Error occurred while checking image consistency: \n{e}")
            raise e

        return response