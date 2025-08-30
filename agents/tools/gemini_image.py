from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from typing import List
import logging
import os

# https://ai.google.dev/gemini-api/docs/image-generation?hl=zh-cn

class GeminiImageGenerator:
    def __init__(
        self,
        api_key: str,
        base_url: str,
    ):
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
            ),
        )

    def __call__(
        self,
        prompt: str,
        image_paths: List[str] = [],
        num_generated_images: int = 1,
        save_path_or_dir: str = None,
        model: str = "gemini-2.5-flash-image-preview",
        **kwargs,
    ):
        """
        Args:
            model: The model to use for image generation

            prompt: The text prompt to guide the image generation

            image_paths: A list of image file paths to use as input for the image generation

            num_generated_images: The number of images to generate
            NOTE: yunwu API does not support the config parameter, so we use a loop to implement it here.

            save_path_or_dir: When generating a single image, provide a file path. When generating multiple images, provide a directory path.
        """


        assert save_path_or_dir is not None
    
        logging.info(f"Calling {model} to generate {num_generated_images} images")

        if num_generated_images > 1:
            os.makedirs(save_path_or_dir, exist_ok=True)

        for i in range(num_generated_images):
            if num_generated_images == 1:
                save_path = save_path_or_dir
            else:
                save_path = os.path.join(save_path_or_dir, f"{i}.png")

            response = self.client.models.generate_content(
                model=model,
                contents=[Image.open(image_path) for image_path in image_paths] + [prompt],
                # config=types.GenerateContentConfig(
                #     candidate_count=num_generated_images,
                # )
            )

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    logging.info(f"Text: {part.text}")
                elif part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(save_path)
                    logging.info(f"Image saved to {save_path}")
