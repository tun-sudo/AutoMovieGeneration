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
from agents.image_generator import ImageGenerator
import shutil


def download_image(url, save_path):
    try:
        # 发送 HTTP GET 请求
        logging.info(f"Downloading image from {url} to {save_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 以二进制写入模式打开文件
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        logging.info(f"Image downloaded successfully to {save_path}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image: {e}")
        raise e




def encode_base64(file_path):
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode('utf-8')
        return f"data:image/jpeg;base64,{base64_string}"




system_prompt_template_select_most_consistent_image = \
"""
You are a professional image quality assessment expert. Your expertise includes identifying facial features and spatial relationships in images, assessing semantic consistency, and evaluating how well generated images match reference images and text descriptions.

**TASK**
Based on the reference image provided by the user, the text description of the generated image, and three candidate generated images, evaluate which candidate image performs best in the following aspects:
1.Character Consistency: Whether the character features (such as appearance, clothing, posture, etc.) in the generated image align with those of the character in the reference image.
2.Spatial Consistency: Whether the relative positions between characters, scene layout, perspective, and other spatial relationships in the generated image are consistent with those in the reference image.
3.Description Accuracy: Whether the generated image accurately reflects the content described in the text (Note: The text description pertains to the generated image itself and is not an editing instruction).

**INPUT**
The user will provide the following content:
- Reference images: These include images of characters or other perspectives, each along with a brief text description. For example, "Reference Image 0: A young girl with long brown hair wearing a red dress." then follow the corresponding image. The index starts from 0.
- Candidate Generated Images: The generated images to be evaluated. For example, "Generated Image 0", then follow a generated image. The index starts from 0.
- Text Description for Generated Image: This describes what the generated image should contain. It is enclosed <IMAGE_DESCRIPTION_START> and <IMAGE_DESCRIPTION_END> tags.

**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Prioritize Character Consistency: Ensure that the characters in the generated image are highly consistent with those in the reference image in terms of visual features (e.g., facial characteristics, hairstyle, clothing, etc.).
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


class SelectMostConsistentImageResponse(BaseModel):
    selected_image_index: int = Field(
        ...,
        description="The index of the selected image."
    )
    reason: str = Field(
        ...,
        description="The reason why the image is the most consistent."
    )


class FrameImageGenerator:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
    ):
        self.gpt4o_client = OpenAI(base_url=base_url, api_key=api_key)

        self.chat_model = init_chat_model(
            model=chat_model,
            model_provider="openai",
            base_url=base_url,
            api_key=api_key,
        )

    @retry
    def __call__(
        self,
        ref_image_path_and_text_pairs: List[Tuple[str, str]],
        guide_prompt: str,
        frame_description: str,
        candidate_save_dir: str,
        best_save_path: str,
    ):
        candidate_image_paths = self.generate_candidate_images(
            ref_image_path_and_text_pairs=ref_image_path_and_text_pairs,
            guide_prompt=guide_prompt,
            save_dir=candidate_save_dir,
        )
        logging.info(f"Candidate images generated at: {candidate_save_dir}")

        logging.info("Selecting the most consistent image...")
        response = self.select_most_consistent_image(
            ref_image_path_and_text_pairs=ref_image_path_and_text_pairs,
            generated_image_paths=candidate_image_paths,
            generated_image_description=frame_description
        )

        best_idx = response.selected_image_index
        logging.info(f"Selected the most consistent image: {best_idx}")
        logging.info(f"Reason: {response.reason}")

        # Save the selected image to the final path
        best_image_path = candidate_image_paths[best_idx]
        shutil.copy(best_image_path, best_save_path)

    def generate_candidate_images(
        self,
        ref_image_path_and_text_pairs: List[Tuple[str, str]],
        guide_prompt: str,
        save_dir: str,
        num_candidates: int = 3,
    ):
        image_paths = []
        image_descriptions = []
        for idx, (ref_image, text) in enumerate(ref_image_path_and_text_pairs):
            image_paths.append(ref_image)
            image_descriptions.append(f"Image {idx}: {text}")

        prompt = "\n".join(image_descriptions) + "\n" + guide_prompt

        self.edit_image(
            prompt=prompt,
            image_paths=image_paths,
            size="1536x1024",
            save_dir=save_dir,
            n=num_candidates,
            background="opaque",
        )

        candidate_image_paths = []
        for img_name in os.listdir(save_dir):
            candidate_image_paths.append(os.path.join(save_dir, img_name))

        return candidate_image_paths


    @retry
    def select_most_consistent_image(
        self,
        ref_image_path_and_text_pairs: List[Tuple[str, str]],
        generated_image_paths: List[str],
        generated_image_description: str,
    ):
        human_content = []

        for idx, (ref_image_url, text) in enumerate(ref_image_path_and_text_pairs):
            human_content.append({
                "type": "text",
                "text": f"Reference Image {idx}: {text}"
            })
            human_content.append({
                "type": "image_url",
                "image_url": {"url": encode_base64(ref_image_url)}
            })


        for idx, gen_image_path in enumerate(generated_image_paths):
            human_content.append({
                "type": "text",
                "text": f"Generated Image {idx}"
            })
            human_content.append({
                "type": "image_url",
                "image_url": {"url": encode_base64(gen_image_path)}
            })

        human_content.append({
            "type": "text",
            "text": human_prompt_template_select_most_consistent_image.format(image_description=generated_image_description)
        })


        parser = PydanticOutputParser(pydantic_object=SelectMostConsistentImageResponse)

        messages = [
            SystemMessage(content=system_prompt_template_select_most_consistent_image.format(format_instructions=parser.get_format_instructions())),
            HumanMessage(content=human_content)
        ]

        chain = self.chat_model | parser

        try:
            response = chain.invoke(messages)

        except Exception as e:
            logging.error(f"Error selecting most consistent image: {e}")
            raise e
        
        return response


    def generate_image(
        self,
        model: str = "gpt-4o-image-vip",
        prompt: str = "",
        size: str = "1024x1024",
        background: str = "auto",
        save_dir: str = None,
        n=1,
    ): 
        response = self.gpt4o_client.images.generate(
            model=model,
            background=background,
            prompt=prompt,
            n=n,
            size=size,
            output_format="url",
        )


        # * Some apis may not support the output_format parameter, so we are trying the following two solutions.
        for idx, data in enumerate(response.data):
            save_path = os.path.join(save_dir, f"{idx}.png")
            if hasattr(data, "url"):
                image_url = data.url
                download_image(image_url, save_path)
            elif hasattr(data, "b64_json"):
                image_data = base64.b64decode(data.b64_json)
                with open(save_dir, "wb") as f:
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
        save_dir: str = None,
        n=1,
    ):
        response = self.gpt4o_client.images.edit(
            model=model,
            prompt=prompt,
            image=[
                open(image_path, "rb") for image_path in image_paths
            ],
            n=n,
            background=background,
            size=size,
            output_format="url",
        )

        # * Some apis may not support the output_format parameter, so we are trying the following two solutions.
        for idx, data in enumerate(response.data):
            save_path = os.path.join(save_dir, f"{idx}.png")
            if hasattr(data, "url"):
                image_url = data.url
                download_image(image_url, save_path)
            elif hasattr(data, "b64_json"):
                image_data = base64.b64decode(data.b64_json)
                with open(save_path, "wb") as f:
                    f.write(image_data)
            else:
                raise ValueError("No image URL or base64 data found in the response.")
