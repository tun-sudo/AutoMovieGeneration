import asyncio
from PIL import Image
from tools.image_generator.nanobanana import NanoBananaImageGenerator
from tools.image_generator.gemini import GeminiImageGenerator


api_key = "sk-7Z55m4gaQXFmvL2jbGqW6CWb0kMiDQe1qkutjeTsxbVTodwY"
base_url = "https://yunwu.ai"

# save_prefix = "example_inputs/images/gemini_output"
# image_generator = GeminiImageGenerator(
#     api_key=api_key,
#     base_url=base_url,
#     model="gemini-2.5-flash-image-preview",
# )

save_prefix = "example_inputs/images/nanobanana_output"
image_generator = NanoBananaImageGenerator(
    api_key=api_key,
    base_url=base_url,
    model="nano-banana",
)

image = asyncio.run(
    image_generator.generate_single_image(
        prompt="A cute magical cat, digital art, watching the stars, running on the beach.",
        reference_images=[Image.open("example_inputs/images/魔法猫咪.png")],
        size="1600x1200",
    )
)
image.save(f"{save_prefix}/single_magical_cat.png")

# images = asyncio.run(
#     image_generator.generate_multiple_images_from_one_prompt(
#         prompt="A cute magical cat, digital art, watching the stars, running on the beach. The cat refers to the cat in the first image.",
#         reference_images=[Image.open("example_inputs/images/魔法猫咪.png")],
#         num_images=2,
#     )
# )
# images.save_all_images(
#     dir_path=f"{save_prefix}/multiple_magical_cat",
#     base_filename="magical_cat",
# )

# images_per_prompt = asyncio.run(
#     image_generator.generate_multiple_images_from_multiple_prompts(
#         prompts=[
#             "A cute magical cat, digital art, watching the stars, running on the beach. The cat refers to the cat in the first image.",
#             "A cute magical cat, digital art, watching the stars, running on the beach. The cat refers to the cat in the first image.",
#         ],
#         reference_images=[
#             [Image.open("example_inputs/images/魔法猫咪.png")],
#             [Image.open("example_inputs/images/小八.png")],
#         ],
#         num_images_per_prompt=2,
#     )
# )
# for idx, images in enumerate(images_per_prompt):
#     images.save_all_images(
#         dir_path=f"{save_prefix}/multiple_magical_cat_{idx}",
#         base_filename="magical_cat",
#     )
