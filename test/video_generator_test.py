import os
import asyncio
import logging
from PIL import Image
from tools.video_generator.veo import VeoVideoGenerator
from tools.video_generator.wan import WanVideoGenerator

logging.basicConfig(level=logging.INFO)


# save_prefix = "example_inputs/videos/veo_output"
# api_key = "sk-RsgJVQohu9e1HBMgdYsy9mQFKs3ue4fZXL2iGMjiiupiViQB"
# base_url = "https://yunwu.ai"
# video_generator = VeoVideoGenerator(
#     api_key=api_key,
#     base_url=base_url,
# )



save_prefix = "example_inputs/videos/wan_output"
video_generator = WanVideoGenerator(
    api_key="9caa641d699a4223b95b7bccebf597c4",
)

prompt = "A cute magical cat, digital art, watching the stars, running on the beach."
reference_image_paths = ["example_inputs/images/nanobanana_output/one_cat.png"]
# reference_image_paths = [
#     "example_inputs/images/nanobanana_output/one_cat.png",
#     "example_inputs/images/nanobanana_output/one_cat.png"
# ]

video = asyncio.run(
    video_generator.generate_single_video(
        prompt=prompt,
        reference_image_paths=reference_image_paths,
    )
)
save_path = f"{save_prefix}/ff2v.mp4" if len(reference_image_paths) == 1 else f"{save_prefix}/flf2v.mp4"
os.makedirs(save_prefix, exist_ok=True)
video.save(save_path)
