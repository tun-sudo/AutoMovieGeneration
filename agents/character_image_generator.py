import os
import logging
from typing import List, Literal
from tenacity import retry
from agents.elements import Character
from agents.tools.gemini_image import GeminiImageGenerator
from agents.tools.gpt4o_image import GPT4oImageGenerator


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



class CharacterImageGenerator:
    def __init__(
        self,
        image_generator_model: Literal["gemini", "gpt4o"],
        api_key: str,
        base_url: str,
    ):
        self.image_generator_model = image_generator_model
        if image_generator_model == "gemini":
            self.image_generator = GeminiImageGenerator(
                api_key=api_key,
                base_url=base_url,
            )
        else:
            self.image_generator = GPT4oImageGenerator(
                api_key=api_key,
                base_url=base_url,
            )


    @retry
    def __call__(
        self,
        character: Character,
        style: str,
        save_dir: str,
    )  -> List[str]:
        """
        Args:
            character: character description
            style: art style
            save_dir: directory to save character images
        """

        logging.info(f"Generating images for character: {character.identifier}")

        character_image_paths = []

        # 1. generate the front view
        prompt_front_view = prompt_template_generate_character_front_view.format(
            features=character.features,
            style=style
        )

        save_path_front_view = os.path.join(save_dir, "front.png")
        if self.image_generator_model == "gemini":
            self.image_generator(
                prompt=prompt_front_view,
                save_path_or_dir=save_path_front_view,
            )
        else:
            # size and background are not supported in GeminiImageGenerator
            self.image_generator(
                prompt=prompt_front_view,
                save_path_or_dir=save_path_front_view,
                size="1024x1536",
                background="transparent",
            )
        character_image_paths.append(save_path_front_view)

        return character_image_paths
