from tools.video_generator.base import BaseVideoGenerator, VideoGeneratorOutput
import logging
from typing import List, Optional
from PIL import Image
import asyncio
import aiohttp
import requests
from tools.video_generator.base import VideoGeneratorOutput, BaseVideoGenerator
from utils.image import image_path_to_b64


class DoubaoDanceVideoGenerator(BaseVideoGenerator):
    def __init__(
        self,
        api_key: str,
        ff2v_model: str = "doubao-seedance-1-0-lite-i2v-250428",
        flf2v_model: str = "doubao-seedance-1-0-lite-i2v-250428",
    ):
        self.api_key = api_key
        self.ff2v_model = ff2v_model
        self.flf2v_model = flf2v_model



    async def generate_single_video(
        self,
        prompt: str,
        reference_image_paths: List[str],
    ) -> VideoGeneratorOutput:
        if len(reference_image_paths) == 1:
            model = self.ff2v_model
        elif len(reference_image_paths) == 2:
            model = self.flf2v_model
        else:
            raise ValueError("reference_image_paths must contain 1 or 2 images.")

        logging.info(f"Calling {model} to generate video...")

        # 1. Create video generation task
        url = "https://yunwu.ai/volc/v1/contents/generations/tasks"

        content = [
            {
                "type": "text",
                "text": prompt + " --rs 480p --rt adaptive --dur 5  --fps 16  --wm false --seed -1  --cf false"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_path_to_b64(reference_image_paths[0])
                },
                "role": "first_frame",
            },
        ]
        if len(reference_image_paths) == 2:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path_to_b64(reference_image_paths[1])
                    },
                    "role": "last_frame",
                }
            )

        payload = {
            "model": model,
            "content": content
        }

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        # 1. Create video generation task
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        response_json = await response.json()
                        task_id = response_json["id"]
            except Exception as e:
                logging.error(f"Error occurred while creating video generation task: {e}")
                logging.info("Retrying in 1 seconds...")
                await asyncio.sleep(1)
                continue
            break
        logging.info(f"Video generation task created. Task ID: {task_id}")


        # 2. Query the video generation task until the video generation is completed
        url = f"https://yunwu.ai/volc/v1/contents/generations/tasks/{task_id}"
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        response_json = await response.json()

            except Exception as e:
                logging.error(f"Error occurred while querying video generation task: {e}")
                logging.info("Retrying in 1 seconds...")
                await asyncio.sleep(1)
                continue

            status = response_json["status"]
            if status == "succeeded":
                video_url = response_json["content"]["video_url"]
                logging.info(f"Video generation succeeded. Video URL: {video_url}")
                break
            elif status == "failed":
                logging.error(f"Video generation failed. Response: {response_json}")
                raise ValueError("Video generation failed.")
            else:
                logging.info(f"Video generation status: {status}. Checking again in 2 seconds...")
                await asyncio.sleep(2)
                continue

        return VideoGeneratorOutput(fmt="url", ext="mp4", data=video_url)

