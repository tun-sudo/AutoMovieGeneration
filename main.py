import asyncio
import logging
from PIL import Image
from tools.video_generator.wan import WanVideoGenerator

logging.basicConfig(level=logging.INFO)

api_key = "9caa641d699a4223b95b7bccebf597c4"
base_url = "https://yunwu.ai"

save_prefix = "example_inputs/videos/veo_output"
video_generator = WanVideoGenerator(
  api_key=api_key,
  base_url=base_url
)

video = asyncio.run(
  video_generator.generate_single_video(
    prompt="A cute magical cat, digital art, watching the stars, running on the beach.",
    reference_images=[Image.open("example_inputs/images/nanobanana_output/multiple_magical_cat/magical_cat_0.png")],
  )
)
video.save(f"{save_prefix}/single_magical_cat.mp4")