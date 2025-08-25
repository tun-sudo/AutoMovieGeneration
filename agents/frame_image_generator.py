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
from agents.image_generator import ImageGenerator


class FrameImageGenerator(ImageGenerator):
    def __init__(
        self,
        base_url: str,
        api_key: str
    ):
        super().__init__(base_url, api_key)


    @retry
    def __call__(
        self,
        ref_image_path_and_text_pairs,
        guide_prompt,
        save_path,
    ):
        image_paths = []
        image_descriptions = []
        for idx, (ref_image, text) in enumerate(ref_image_path_and_text_pairs):
            image_paths.append(ref_image)
            image_descriptions.append(f"Image {idx}: {text}")

        prompt = "\n".join(image_descriptions) + "\n" + guide_prompt

        self.edit_image(
            prompt=prompt,
            image_paths=image_paths,
            size="1536x1024",
            save_path=save_path,
        )
