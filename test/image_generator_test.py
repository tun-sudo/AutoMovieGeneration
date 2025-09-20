import os
import asyncio
from PIL import Image
from tools.image_generator.nanobanana import NanoBananaImageGenerator
from tools.image_generator.gemini import GeminiImageGenerator
from tools.image_generator.gpt4o import GPT4oImageGenerator
from tools.image_generator.doubao_seedream import DoubaoSeedreamImageGenerator
import logging

logging.basicConfig(level=logging.INFO)

api_key = "sk-7Z55m4gaQXFmvL2jbGqW6CWb0kMiDQe1qkutjeTsxbVTodwY"
base_url = "https://yunwu.ai"

# prompt = r"生成一张大树的图片，电影真实风格"

# prompt = "A cute magical cat, digital art, watching the stars, running on the beach."
# reference_image_paths = ["example_inputs/images/魔法猫咪.png"]

# prompt = "Two cute cats, digital art, watching the stars, running on the beach. One refers to the first image, the other refers to the second image."
prompt = "Two cute cats are fighting with each other. One fights like a brave knight, the other fights like a ninja. The brave knight cat wears shining armor and holds a sword and shield, while the ninja cat wears a black ninja outfit and holds sharp throwing stars. They are in a dynamic battle pose, with intense expressions on their faces. The background is a dramatic battlefield with swirling dust and flying debris, capturing the excitement of their epic showdown. Digital art, high detail, vibrant colors. One cat refers to the first image, the other cat refers to the second image."
# prompt = "生成3张女孩和奶牛玩偶在游乐园开心地坐过山车的图片，涵盖早晨、中午、晚上。"
# reference_image_paths = [
#     "example_inputs/images/seedream4_imagesToimages_1.png",
#     "example_inputs/images/seedream4_imagesToimages_2.png"
# ]

reference_image_paths = [
    "example_inputs/images/魔法猫咪.png",
    "example_inputs/images/小八.png"
]

# save_prefix = "example_inputs/images/gpt4o_output"
# image_generator = GPT4oImageGenerator(
#     api_key=api_key,
#     base_url=base_url,
#     model="gpt-image-1-all",
# )

# save_prefix = "example_inputs/images/gemini_output"
# image_generator = GeminiImageGenerator(
#     api_key=api_key,
#     base_url=base_url,
#     model="gemini-2.5-flash-image-preview",
# )

save_prefix = "example_inputs/images/nanobanana_output"
image_generator = NanoBananaImageGenerator(
    api_key=api_key,
    base_url=base_url,
    model="nano-banana",
)

# save_prefix = "example_inputs/images/doubao_seedream_output"
# image_generator = DoubaoSeedreamImageGenerator(
#     api_key=api_key,
#     model="doubao-seedream-4-0-250828",
# )

image = asyncio.run(
    image_generator.generate_single_image(
        prompt=prompt,
        reference_image_paths=reference_image_paths,
        size="1600x1024",
    )
)
os.makedirs(save_prefix, exist_ok=True)
# save_path = f"{save_prefix}/tree.png"
# save_path = f"{save_prefix}/girl_and_cow.png"
# save_path = f"{save_prefix}/magical_cat.png"
save_path = f"{save_prefix}/cat_fight.png"
image.save(save_path)
