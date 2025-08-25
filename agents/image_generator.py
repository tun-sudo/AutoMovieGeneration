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


def download_image(url, save_path):
    try:
        # 发送 HTTP GET 请求
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 以二进制写入模式打开文件
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        logging.info(f"Image downloaded successfully to {save_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image: {e}")




def encode_base64(file_path):
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"






class ImageGenerator:
    def __init__(
        self,
        base_url: str = "https://yunwu.ai/v1",
        api_key: str = "sk-bapMAyiji5uA2O91yTv6iBIXmp3e0JsMWByxgzHkzOd9jRIF",
    ):
        self.gpt4o_client = OpenAI(base_url=base_url, api_key=api_key)


    def generate_frame_image(
        self,
        ref_image_text_pairs,
        prompt,
        save_path,
    ):
        image_paths = []
        image_descriptions = []
        for idx, (ref_image, text) in enumerate(ref_image_text_pairs):
            image_paths.append(ref_image)
            image_descriptions.append(f"Image {idx}: {text}")

        prompt = "\n".join(image_descriptions) + "\n" + prompt
        print(prompt)

        self.edit_image(
            prompt=prompt,
            image_paths=image_paths,
            size="1536x1024",
            save_path=save_path,
        )


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
        if hasattr(response.data[0], "url"):
            image_url = response.data[0].url
            download_image(image_url, save_path)
        elif hasattr(response.data[0], "b64_json"):
            image_data = base64.b64decode(response.data[0].b64_json)
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

        if hasattr(response.data[0], "url"):
            image_url = response.data[0].url
            download_image(image_url, save_path)
        elif hasattr(response.data[0], "b64_json"):
            image_data = base64.b64decode(response.data[0].b64_json)
            with open(save_path, "wb") as f:
                f.write(image_data)
        else:
            raise ValueError("No image URL or base64 data found in the response.")

