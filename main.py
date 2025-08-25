from agents.video_generation_pipeline import VideoGenerationPipeline
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Pipeline initialized.")

base_url: str = "https://yunwu.ai/v1"
api_key: str = "sk-bapMAyiji5uA2O91yTv6iBIXmp3e0JsMWByxgzHkzOd9jRIF"

script_path = r"example_inputs/短视频剧本/0-套路.txt"
working_dir = r".working_dir/0-套路"
style = "写实主义，高清，电影感"



script = open(script_path, "r", encoding="utf-8").read()

pipeline = VideoGenerationPipeline(
   base_url=base_url,
   api_key=api_key,
   working_dir=working_dir
)

pipeline(script, style=style)
