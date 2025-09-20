import os
import logging
import asyncio
import aiohttp
from io import BytesIO
from PIL import Image
from typing import List, Optional
import requests
from utils.image import pil_to_b64
import json
import time
from tools.image_generator.base import BaseImageGenerator, ImageGeneratorOutput
from tenacity import retry, stop_after_attempt

# https://yunwu.apifox.cn/api-341952136
# only for fal-ai nano-banana model

class NanoBananaImageGenerator(BaseImageGenerator):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "nano-banana",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model


    async def _generate_single_image(
        self,
        prompt,
        reference_images_b64: List[str] = [],
    ):
        url = f"{self.base_url}/fal-ai/nano-banana/edit"
        payload = {
            "prompt": prompt,
            "image_urls": reference_images_b64,
            "num_images": 1,
            "output_format": "png",
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }


        # 1. Create image generation task
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        response_json = await response.json()
                        logging.debug("Response JSON: " + str(response_json))
                        task_id = response_json.get("request_id")
                        logging.info(f"Image generation task created with ID: {task_id}")
            except Exception as e:
                logging.error(f"Error occurred while creating image generation task: {e}")
                await asyncio.sleep(2)
                continue
            break


        # 2. Query the image generation task until the image generation is completed
        url = f"{self.base_url}/fal-ai/auto/requests/{task_id}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        response_json = await response.json()
                        logging.debug("Response JSON: " + str(response_json))
            except Exception as e:
                logging.error(f"Error occurred while querying image generation task: {e}")
                await asyncio.sleep(2)
                continue

            status = response_json.get("status", None)
            if status:
                logging.info(f"Image generation task status: {status}")
                continue

            if "images" not in response_json:
                raise ValueError(f"No image generated. The response is: {response_json}")

            image_url = response_json["images"][0]["url"]
            return image_url

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
        
        logging.info(f"Calling {self.model} to generate image.")

        reference_images = [Image.open(path) for path in reference_image_paths]

        image_url = await self._generate_single_image(
            prompt=prompt,
            reference_images_b64=[pil_to_b64(image) for image in reference_images],
        )


        # The area of the generated image must be nearly or equal to 1024*1024
        # We can control the aspect ratio by redrawing the image on a white board
        if size is not None:
            #  redraw the image on a white board
            target_width, target_height = map(int, size.split("x"))
            blank_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))

            reference_images_b64 = [image_url, pil_to_b64(blank_image)]
            prompt = "Redraw the content of Figure 1 onto Figure 2, add content to Figure 1 to fit the aspect ratio of Figure 2, completely clear the content of Figure 2, and only retain the aspect ratio of Figure 2. The generated image is free from any black borders, white borders, frames, or similar elements. If the content is not enough, you can add more details to the content of Figure 1, but the style and content should be consistent with Figure 1."
            image_url = await self._generate_single_image(
                prompt=prompt,
                reference_images_b64=reference_images_b64,
            )
            logging.info(f"Redraw completed.")


        return ImageGeneratorOutput(fmt="url", ext="png", data=image_url)

