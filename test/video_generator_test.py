import os
import asyncio
import logging
from PIL import Image
from tools.video_generator.veo import VeoVideoGenerator
from tools.video_generator.wan import WanVideoGenerator
from tools.video_generator.doubao_seedance import DoubaoDanceVideoGenerator

logging.basicConfig(level=logging.INFO)


# save_prefix = "example_inputs/videos/veo_output"
# api_key = "sk-RsgJVQohu9e1HBMgdYsy9mQFKs3ue4fZXL2iGMjiiupiViQB"
# video_generator = VeoVideoGenerator(
#     api_key=api_key,
# )



# save_prefix = "example_inputs/videos/wan_output"
# video_generator = WanVideoGenerator(
#     api_key="9caa641d699a4223b95b7bccebf597c4",
# )


save_prefix = "example_inputs/videos/doubao_output"
video_generator = DoubaoDanceVideoGenerator(
    api_key="sk-RsgJVQohu9e1HBMgdYsy9mQFKs3ue4fZXL2iGMjiiupiViQB",
)


# prompt = "The transformation process from a seed to a big tree. Natural transition, high detail, vibrant colors."
prompt = "镜头切换。第一个镜头为一颗种子在土壤中萌发，慢慢长出嫩芽。第二个镜头为大树的树干逐渐长高，树枝伸展。自然过渡，高细节，色彩鲜艳。"
reference_image_paths = [
    r"example_inputs\images\nanobanana_output\seed.png",
    r"example_inputs\images\nanobanana_output\tree.png"
]


reference_image_paths = [
    r"example_inputs\images\nanobanana_output\seed.png",
    r"example_inputs\images\nanobanana_output\tree.png"
]

video = asyncio.run(
    video_generator.generate_single_video(
        prompt=prompt,
        reference_image_paths=reference_image_paths,
    )
)
# save_path = f"{save_prefix}/ff2v.mp4" if len(reference_image_paths) == 1 else f"{save_prefix}/flf2v.mp4"
# save_path = f"{save_prefix}/seed_to_tree_ff2v.mp4"
save_path = f"{save_prefix}/seed_to_tree_flf2v.mp4"
os.makedirs(save_prefix, exist_ok=True)
video.save(save_path)
