import os
import logging
import asyncio
from io import BytesIO
from PIL import Image
from typing import List, Optional
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt

from tools.image_generator.base import BaseImageGenerator, ImageGeneratorOutput

# https://ai.google.dev/gemini-api/docs/image-generation?hl=zh-cn

class GeminiImageGenerator(BaseImageGenerator):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        api_version: str = "v1beta",
        model: str = "gemini-2.5-flash-image-preview",
    ):
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                api_version=api_version,
            ),
        )
        self.model = model



    async def _generate_single_image(
        self,
        prompt: str,
        reference_images: List[Image.Image] = [],
    ) -> ImageGeneratorOutput:

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=reference_images + [prompt],
        )

        image = None
        text = ""
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text += part.text
            elif part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))

        if image is None:
            raise ValueError("No image generated. The response text is: " + text)

        return image


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

        reference_images = [Image.open(path) for path in reference_image_paths]

        image = await self._generate_single_image(
            prompt=prompt,
            reference_images=reference_images,
        )


        # The area of the generated image must be nearly or equal to 1024*1024
        # We can control the aspect ratio by redrawing the image on a white board
        if size is not None:
            target_width, target_height = map(int, size.split("x"))
            current_width, current_height = image.size

            # If the aspect ratio is different, redraw the image on a white board
            if current_height * target_width != current_width * target_height:
                logging.info(f"The aspect ratio of the generated image {current_width}x{current_height} is different from the target size {target_width}x{target_height}. Redrawing the image on a white board.")

                blank_image = Image.new("RGB", (target_width, target_height), (255, 255, 255))

                reference_images = [image, blank_image]
                prompt = "Redraw the content of Figure 1 onto Figure 2, add content to Figure 1 to fit the aspect ratio of Figure 2, completely clear the content of Figure 2, and only retain the aspect ratio of Figure 2. The generated image is free from any black borders, white borders, frames, or similar elements. If the content is not enough, you can add more details to the content of Figure 1, but the style and content should be consistent with Figure 1."
                image = await self._generate_single_image(
                    prompt=prompt,
                    reference_images=reference_images,
                )
                logging.info(f"Redraw completed. The size of the new image is {image.size[0]}x{image.size[1]}.")

        return ImageGeneratorOutput(fmt="pil", ext="png", data=image)

