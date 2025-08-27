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


prompt_template_generate_character_front_view = \
"""
Generate a full-body, front-view portrait of a character based on the following description, with an empty background. The character should be centered in the image, occupying most of the frame. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression.
features: {features}
style: {style}
"""

prompt_template_generate_character_side_view = \
"""
Generate a full-body, side-view portrait of a character based on the following description and the given front-view portrait, with an empty background. The character should be centered in the image, occupying most of the frame. Facing left. Standing with arms relaxed at sides. Natural expression.
features: {features}
style: {style}
"""


prompt_template_generate_character_back_view = \
"""
Generate a full-body, back-view portrait of a character based on the following description and the given front-view portrait, with an empty background. The character should be centered in the image, occupying most of the frame. No facial features visible. Standing with arms relaxed at sides.
features: {features}
style: {style}
"""



class CharacterImageGenerator(ImageGenerator):
    def __init__(
        self,
        base_url: str = "https://yunwu.ai/v1",
        api_key: str = "sk-bapMAyiji5uA2O91yTv6iBIXmp3e0JsMWByxgzHkzOd9jRIF",
    ):
        super().__init__(base_url=base_url, api_key=api_key)

    @retry
    def __call__(
        self,
        character: Character,
        style: str,
        save_dir: str,
    ):
        # 1. generate the front view
        prompt_front_view = prompt_template_generate_character_front_view.format(
            features=character.features,
            style=style
        )
        save_path_front_view = os.path.join(save_dir, "front.png")
        self.generate_image(
            prompt=prompt_front_view,
            size="1024x1536",
            background="transparent",
            save_dir=save_path_front_view,
        )

        # 2. then create the side view and back view based on the front view.
        # prompt_side_view = prompt_template_generate_character_side_view.format(
        #     features=character.features,
        #     style=style
        # )
        # save_path_side_view = os.path.join(save_dir, "side.png")
        # self.edit_image(
        #     prompt=prompt_side_view,
        #     image_paths=[save_path_front_view],
        #     size="1024x1536",
        #     background="transparent",
        #     save_path=save_path_side_view
        # )

        # prompt_back_view = prompt_template_generate_character_back_view.format(
        #     features=character.features,
        #     style=style
        # )
        # save_path_back_view = os.path.join(save_dir, "back.png")
        # self.edit_image(
        #     prompt=prompt_back_view,
        #     image_paths=[save_path_front_view],
        #     size="1024x1536",
        #     background="transparent",
        #     save_path=save_path_back_view
        # )
