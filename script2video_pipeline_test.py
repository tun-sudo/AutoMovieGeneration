import asyncio
from pipelines.script2video_pipeline import Script2VideoPipeline


config_path = r"configs/script2video.yaml"
script_path = r"example_inputs\shot_video_scripts\0-套路.txt"
working_dir = ".working_dir/script2video_pipeline_test/0-套路"
style = "二次元"

pipeline = Script2VideoPipeline.init_from_config(
    config_path=config_path,
    working_dir=working_dir,
)

script = open(script_path, "r", encoding="utf-8").read()
asyncio.run(pipeline(script=script, style=style))
