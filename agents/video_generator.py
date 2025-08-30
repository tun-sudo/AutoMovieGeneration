import http.client
import json
import base64
import time
import requests
import os
from agents.utils.image import image_to_base64_with_mime, download_image
from agents.utils.video import download_video
import logging


class VideoGenerator:
    def __init__(
        self,
        base_url: str,
        api_key: str,
    ):
        self.base_url = base_url
        self.api_key = api_key

    def __call__(
        self,
        prompt: str = "",
        image_paths: list = [],
        save_path: str = None,
    ):
        logging.info(f"Generating video ...")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"First Frame: {image_paths[0]}")
        if len(image_paths) > 1:
            logging.info(f"Last Frame: {image_paths[-1]}")

        conn = http.client.HTTPSConnection("yunwu.ai")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        images = []
        for image_path in image_paths:
            if image_path:
                images.append(image_to_base64_with_mime(image_path))
        payload = json.dumps({
            "prompt": prompt,
            "model": "veo2-fast-frames" if len(images) == 2 else "veo3-fast-frames",
            "images": images,
            "enhance_prompt": True
        })

        conn.request("POST", "/v1/video/create", payload, headers)
        res = conn.getresponse()

        response_data = json.loads(res.read().decode("utf-8"))
        task_id = response_data["id"]
        boundary = ''
        payload = ''
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
        }

        while True:
            conn.request("GET", f"/v1/video/query?id={task_id}", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            if data["status"] == "completed":
                logging.info(f"Video generation completed successfully")
                video_url = data["video_url"]
                download_video(video_url, save_path)
                break
            elif data["status"] == "failed":
                logging.error(f"Video generation failed: \n{data}")
                break
            else:
                logging.info(f"Video generation status: {data['status']}, waiting 5 seconds...")
                time.sleep(5)
                continue
