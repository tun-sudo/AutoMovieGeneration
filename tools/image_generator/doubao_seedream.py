import os
import logging
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image
from typing import List, Optional
from tenacity import retry, stop_after_attempt
import requests

from tools.image_generator.base import BaseImageGenerator, ImageGeneratorOutput

from utils.image import image_path_to_b64

# https://yunwu.apifox.cn/api-347960869

class DoubaoSeedreamImageGenerator(BaseImageGenerator):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://yunwu.ai/v1/images/generations",
        model: str = "doubao-seedream-4-0-250828",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model


    @retry(
        stop=stop_after_attempt(3),
        after=lambda retry_state: logging.warning(f"Retrying generate_single_image due to error: {retry_state.outcome.exception()}"),
    )
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] = [],
        size: Optional[str] = None,
    ) -> ImageGeneratorOutput:
        """
        size: [1024x1024, 4096x4096]
        ratio: [1/16, 16]
        """

        image = [
            image_path_to_b64(path, mime=True) for path in reference_image_paths
        ]

        payload = {
            "model": self.model,
            "prompt": prompt,
            "image": image,   # NOTE: the key is image, not images
            "sequential_image_generation": "disabled",  # "auto" or "disabled"
            # "sequential_image_generation_options": {
            #     "max_images": 1
            # },
            "response_format": "url",
            "size": size if size is not None else "1024x1024",
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # response = requests.post(self.base_url, json=payload, headers=headers)
        # response_json = response.json()
        # print(response_json)

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload, headers=headers) as response:
                response_json = await response.json()

        data = response_json['data'][0]['url']
        return ImageGeneratorOutput(fmt="url", ext="png", data=data)
