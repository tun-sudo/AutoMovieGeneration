import asyncio
import logging
from PIL import Image
from tools.video_generator.veo import VeoVideoGenerator

logging.basicConfig(level=logging.INFO)

api_key = "sk-RsgJVQohu9e1HBMgdYsy9mQFKs3ue4fZXL2iGMjiiupiViQB"
base_url = "https://yunwu.ai"

save_prefix = "example_inputs/videos/veo_output"
video_generator = VeoVideoGenerator(
    api_key=api_key,
    base_url=base_url,
)

video = asyncio.run(
    video_generator.generate_single_video(
        prompt="A cute magical cat, digital art, watching the stars, running on the beach.",
        reference_images=[Image.open("example_inputs/images/nanobanana_output/multiple_magical_cat/magical_cat_0.png")],
    )
)
video.save(f"{save_prefix}/single_magical_cat.mp4")


# videos = asyncio.run(
#     video_generator.generate_multiple_videos_from_multiple_prompts(
#         prompts=[
#             "A cute magical cat, digital art, watching the stars, running on the beach.",
#             "A cute magical cat, digital art, watching the stars, running on the beach.",
#         ],
#         reference_images=[
#             [Image.open("example_inputs/images/nanobanana_output/multiple_magical_cat_0/magical_cat_0.png")],
#             [Image.open("example_inputs/images/nanobanana_output/multiple_magical_cat_1/magical_cat_1.png")],
#         ],
#         num_videos_per_prompt=1,
#     )
# )
# for idx, video in enumerate(videos):
#     video.save_all_videos(
#         dir_path=f"{save_prefix}/multiple_magical_cat_{idx}",
#         base_filename="magical_cat_running_on_beach",
#     )
