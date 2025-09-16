import logging
import json
from pipelines.novel2movie_pipeline import Novel2MoviePipeline
from pipelines.script2video_pipeline import Script2VideoPipeline
logging.basicConfig(level=logging.INFO)
logging.info("Pipeline initialized.")

pipeline = Script2VideoPipeline.init_from_config(
    config_path="configs/script2video.yaml",
    working_dir=".working_dir/script2video",
)


