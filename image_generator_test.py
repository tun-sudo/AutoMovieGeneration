import json
import logging
from agents.tools.gpt4o_image import GPT4oImageGenerator
from agents.tools.gemini_image import GeminiImageGenerator

logging.basicConfig(level=logging.INFO)


# image_generator = GPT4oImageGenerator(
#     api_key="sk-luBuLGKLIdPoN78dJ3uc9Ra9n1QG2fypirTyRa7qs2WD8OOs",
#     base_url="https://yunwu.ai/v1",
# )

# image_generator(
#     model="gpt-image-1-all",
#     prompt="Fig 1 depicts a cartoon cat. Draw a gorgeous image of a river made of white owl feathers, snaking its way through a serene winter landscape. The cat sits on the riverbank, watching the water flow.",
#     image_paths=["example_inputs/images/cat.png"],
#     num_generated_images=2,
#     save_path_or_dir="example_inputs/images/cat_river_gpt4o",
#     size="1536x1024",
# )


image_generator = GeminiImageGenerator(
    api_key="sk-RsgJVQohu9e1HBMgdYsy9mQFKs3ue4fZXL2iGMjiiupiViQB",
    base_url="https://yunwu.ai",
)

image_generator(
    model="gemini-2.5-flash-image-preview",
    prompt="Draw a gorgeous image of a river made of white owl feathers, snaking its way through a serene winter landscape. Size: 1536x1024",
    num_generated_images=1,
    save_path_or_dir="example_inputs/images/river.png"
)

# image_generator(
#     model="gemini-2.5-flash-image-preview",
#     prompt="Fig 1 depicts a cartoon cat. Fig 2 depicts a river. Create an image: The cat sits on the riverbank, watching the water flow.",
#     image_paths=["example_inputs/images/魔法猫咪.png", "example_inputs/images/river.png"],
#     num_generated_images=1,
#     save_path_or_dir="example_inputs/images/cat_river.png",
# )
