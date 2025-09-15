import logging
import json
from agents.video_generation_pipeline import VideoGenerationPipeline

logging.basicConfig(level=logging.INFO)
logging.info("Pipeline initialized.")


script_path = r"example_inputs/短视频剧本/0-套路.txt"
working_dir = r".working_dir/0-套路"
style = "写实主义，高清，电影感"


script = open(script_path, "r", encoding="utf-8").read()



with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
pipeline = VideoGenerationPipeline(
    config=config,
    working_dir=working_dir
)

pipeline(script, style=style)
