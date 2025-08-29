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



def download_image(url, save_path):
    try:
        # 发送 HTTP GET 请求
        logging.info(f"Downloading image from {url} to {save_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 以二进制写入模式打开文件
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        logging.info(f"Image downloaded successfully to {save_path}")
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        raise e




def encode_base64(file_path):
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"




class CharacterImageGenerator:
    def __init__(
        self,
        base_url: str = "https://yunwu.ai/v1",
        api_key: str = "sk-7Z55m4gaQXFmvL2jbGqW6CWb0kMiDQe1qkutjeTsxbVTodwY",
    ):
        self.gpt4o_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

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
            save_path=save_path_front_view,
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



    def generate_image(
        self,
        model: str = "gpt-4o-image-vip",
        prompt: str = "",
        size: str = "1024x1024",
        background: str = "auto",
        save_path: str = None,
    ): 
        response = self.gpt4o_client.images.generate(
            model=model,
            background=background,
            prompt=prompt,
            n=1,
            size=size,
            output_format="url",
        )

        # * Some apis may not support the output_format parameter, so we are trying the following two solutions.
        data = response.data[0]
        if hasattr(data, "url"):
            image_url = data.url
            download_image(image_url, save_path)
        elif hasattr(data, "b64_json"):
            image_data = base64.b64decode(data.b64_json)
            with open(save_path, "wb") as f:
                f.write(image_data)
        else:
            raise ValueError("No image URL or base64 data found in the response.")


    def edit_image(
        self,
        model: str = "gpt-4o-image-vip",  # gpt-4o-image-vip gpt-image-1-all
        prompt: str = "",
        image_paths: List[str] = [],
        size: str = "1024x1024",
        background: str = "auto",
        save_path: str = None,
    ):
        response = self.gpt4o_client.images.edit(
            model=model,
            prompt=prompt,
            image=[
                open(image_path, "rb") for image_path in image_paths
            ],
            n=1,
            background=background,
            size=size,
            output_format="url",
        )

        # * Some apis may not support the output_format parameter, so we are trying the following two solutions.
        data = response.data[0]
        if hasattr(data, "url"):
            image_url = data.url
            download_image(image_url, save_path)
        elif hasattr(data, "b64_json"):
            image_data = base64.b64decode(data.b64_json)
            with open(save_path, "wb") as f:
                f.write(image_data)
        else:
            raise ValueError("No image URL or base64 data found in the response.")
