import os
import logging
from typing import List, Literal, Tuple
from tenacity import retry
from agents.tools.gemini_image import GeminiImageGenerator
from agents.tools.gpt4o_image import GPT4oImageGenerator


system_prompt_template_select_most_consistent_image = \
"""
You are a professional visual assessment expert. Your expertise includes identifying Character Consistency and Spatial Relationships in images, assessing semantic consistency, and evaluating how well generated images match reference images and text descriptions.

**TASK**
Based on the reference image provided by the user, the text description of the generated image, and three candidate generated images, evaluate which candidate image performs best in the following aspects:

1.Character Consistency: Whether the character features (a. gender b.ethnicity, c.age, d.facial features, e.body shape, f.outlook, g. hairstyle) in the generated image align with those of the character in the reference image.

2.Spatial Consistency: Whether the relative positions between characters (eg. Character A is on the left, character B is on the right), scene layout, perspective, and other spatial relationships in the generated image are consistent with those in the reference image.

3.Description Accuracy: Whether the generated image accurately 
reflects the content described in the text (Note: The text description pertains to the generated image itself and is not an editing instruction).

**INPUT**
The user will provide the following content:
- Reference images: These include images of characters or other perspectives, each along with a brief text description. For example, "Reference Image 0: A young girl with long brown hair wearing a red dress." then follow the corresponding image. The index starts from 0.
- Candidate Generated Images: The generated images to be evaluated. For example, "Generated Image 0", then follow a generated image. The index starts from 0.
- Text Description for Generated Image: This describes what the generated image should contain. It is enclosed within <IMAGE_DESCRIPTION_START> and <IMAGE_DESCRIPTION_END> tags.

**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Prioritize Character Consistency: Ensure that the characters in the generated image are highly consistent with those in the reference image in terms of visual features (e.g., a. gender b.ethnicity, c.age, d.facial features, e.body shape, f.outlook, g. hairstyle etc.).

2. Focus on Spatial Consistency: Verify whether the relative positions of characters, object arrangements, and perspectives align logically with the reference image (e.g., if Character A is on the left and Character B is on the right in the reference image, the generated image should not reverse this).

3.Strictly Compare with Text Description: The generated image must adhere to key elements in the text description (e.g., actions, scenes, objects, etc.), while disregarding parts related to editing instructions (as the input description reflects the expected outcome rather than directives).

4. If multiple images partially meet the criteria, select the one with the highest overall consistency; if none are ideal, choose the relatively best option and explain its shortcomings.

5. Avoid subjective preferences; base all analysis on objective comparisons.
"""


human_prompt_template_select_most_consistent_image = \
"""
<IMAGE_DESCRIPTION_START>
{image_description}
<IMAGE_DESCRIPTION_END>
"""



class FrameCandidateImagesGenerator:
    def __init__(
        self,
        image_generator_model: Literal["gemini", "gpt4o"],
        base_url: str,
        api_key: str,
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
        ref_image_path_and_text_pairs: List[Tuple[str, str]],
        guide_prompt: str,
        save_dir: str,
        num_images: int = 3
    ):
        logging.info("Generating candidate images...")

        try:
            image_paths = []
            image_descriptions = []
            for idx, (ref_image, text) in enumerate(ref_image_path_and_text_pairs):
                image_paths.append(ref_image)
                image_descriptions.append(f"Image {idx}: {text}")

            prompt = "\n".join(image_descriptions) + "\n" + guide_prompt

            if self.image_generator_model == "gemini":
                self.image_generator(
                    prompt=prompt,
                    image_paths=image_paths,
                    save_path_or_dir=save_dir,
                    num_generated_images=num_images,
                )
            else:
                self.image_generator(
                    prompt=prompt,
                    image_paths=image_paths,
                    save_path_or_dir=save_dir,
                    num_generated_images=num_images,
                    size="1536x1024",
                    background="opaque",
                )

            candidate_image_paths = []
            for img_name in os.listdir(save_dir):
                candidate_image_paths.append(os.path.join(save_dir, img_name))

            return candidate_image_paths

        except Exception as e:
            logging.error(f"Error generating candidate images: {e}")
            raise e
