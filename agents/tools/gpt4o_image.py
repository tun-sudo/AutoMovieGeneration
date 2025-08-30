import os
import base64
from openai import OpenAI
from agents.utils.image import download_image
from typing import List
import logging

# https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1&api=image

class GPT4oImageGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)


    def __call__(
        self,
        prompt: str = None,
        image_paths: List[str] = [],
        num_generated_images: int = 1,
        save_path_or_dir: str = None,
        model: str = "gpt-image-1-all",
        **kwargs,  # background, size, ...
    ):
        """
        Args:
            model: The model to use for image generation

            prompt: The text prompt to guide the image generation

            image_paths: A list of image file paths to use as input for the image generation

            num_generated_images: The number of images to generate

            save_path_or_dir: When generating a single image, provide a file path. When generating multiple images, provide a directory path.

            background: Must be one of `transparent`, `opaque` or `auto` (default value). When `auto` is used, the model will automatically determine the best background for the image.

            size: Must be one of `1024x1024`, `1536x1024` (landscape), `1024x1536` (portrait), or `auto` (default value) for `gpt-image-1`
        """

        logging.info(f"Calling {model} to generate {num_generated_images} images")

        if len(image_paths) == 0:
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                n=num_generated_images,
                response_format="b64_json",
                **kwargs,
            )
            
        else:
            # For `gpt-image-1`, each image should be a `png`, `webp`, or `jpg` file less than 50MB. You can provide up to 16 images.

            # check image file types and sizes
            for image_path in image_paths:
                if not image_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    raise ValueError(f"Invalid image file type: {image_path}")
                if os.path.getsize(image_path) > 50 * 1024 * 1024:
                    raise ValueError(f"Image file too large: {image_path}")

            if len(image_paths) > 16:
                raise ValueError("Too many images provided. Please provide up to 16 images.")

            response = self.client.images.edit(
                model=model,
                prompt=prompt,
                image=[
                    open(image_path, "rb") for image_path in image_paths
                ],
                n=num_generated_images,
                response_format="b64_json",
                **kwargs,
            )

        if num_generated_images > 1:
            os.makedirs(save_path_or_dir, exist_ok=True)

        for idx, data in enumerate(response.data):
            if num_generated_images == 1:
                save_path = save_path_or_dir
            else:
                save_path = os.path.join(save_path_or_dir, f"{idx}.png")

            if hasattr(data, "b64_json"):
                image_data = base64.b64decode(data.b64_json)
                with open(save_path, "wb") as f:
                    f.write(image_data)
            elif hasattr(data, "url"):
                # some api return url only, so we try to download the image
                image_url = data.url
                download_image(image_url, save_path)
            else:
                raise ValueError("No image URL or base64 data found in the response.")

            logging.info(f"Image saved to {save_path}")
