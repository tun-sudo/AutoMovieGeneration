from agents.video_generation_pipeline import VideoGenerationPipeline
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Pipeline initialized.")

base_url: str = "https://yunwu.ai/v1"
api_key: str = "sk-bapMAyiji5uA2O91yTv6iBIXmp3e0JsMWByxgzHkzOd9jRIF"



script = \
"""
小偷在外面敲窗户，老王开门。
老王一瞪眼：干嘛? 
小偷：哎，你门没关，老王一看，果然没关。
老王激动：谢谢兄弟，好人！
小偷：没事，路过提个醒。
老王关门，小偷溜走。
小偷窃喜：得亏试探下，果然有人！
老王庆幸：赶紧拿点东西走人，干这行太危险，差点被抓！
"""

pipeline = VideoGenerationPipeline(
   base_url=base_url,
   api_key=api_key,
   working_dir=r".working_dir/0-套路"
)

pipeline(script, style="写实主义，高清，电影感")
