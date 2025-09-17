import os
import shutil
import json
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List
from PIL import Image
from tenacity import retry

from pipelines.base import BasePipeline
from tools.image_generator.base import ImageGeneratorOutput
from tools.video_generator.base import VideoGeneratorOutput, BaseVideoGenerator
from components.character import CharacterInScene
from components.shot import Shot


class Script2VideoPipeline(BasePipeline):
    """
    ----{working_dir}
        |---- characters
        |       |---- character_registry.json
        |       |---- {character_identifier_in_scene}.json
        |       |---- {character_identifier_in_scene}.png
        |---- shots
        |       |----reference_image_paths
        |       |       |----{shot_idx}_first_frame_reference.json
        |       |       |----{shot_idx}_last_frame_reference.json
        |       |----frame_candidates
        |       |       |----shot_{shot_idx}_first_frame
        |       |               |----0.png
        |       |       |----shot_{shot_idx}_last_frame
        |       |               |----0.png
        |       |----{shot_idx}.json
        |       |----{shot_idx}_first_frame.png
        |       |----{shot_idx}_last_frame.png
        |       |----{shot_idx}_video.mp4
    """

    async def __call__(
        self,
        script: str,
        style: str,
        character_registry: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ):
        """
        Args:
            script (str): The input script for video generation.

            character_registry: Optional dictionary mapping character names to their descriptions.
            For example,
            {
                "Alice": [
                    {"path": "path/to/alice1.png", "description": "A front-view portrait of Alice."},
                    {"path": "path/to/alice2.png", "description": "A side-view portrait of Alice."},
                ],
                "Bob": [
                    {"path": "path/to/bob1.png", "description": "A front-view portrait of Bob."},
                ],
            }
        """

        print("="*60)
        print("🎬 STARTING VIDEO GENERATION PIPELINE")
        print("="*60)

        if character_registry is None:
            print("⭕ Phase 0: Extract characters and generate portraits...")
            start_time_0 = time.time()
            character_registry = await self._extract_characters_and_generate_portraits(
                script=script,
                style=style,
            )
            end_time_0 = time.time()
            print(f"✅ Phase 0 completed in {end_time_0 - start_time_0:.2f} seconds.")



        print("⭕ Phase 1: Design storyboard, generate frames and generate shots...")
        start_time_1 = time.time()
        await self._design_storyboard_and_generate_shots(
            script=script,
            character_registry=character_registry,
        )
        end_time_1 = time.time()
        print(f"✅ Phase 1 completed in {end_time_1 - start_time_1:.2f} seconds.")



    async def _extract_characters_and_generate_portraits(
        self,
        script: str,
        style: str,
    ):
        working_dir_characters = os.path.join(self.working_dir, "characters")
        os.makedirs(working_dir_characters, exist_ok=True)
        print(f"🗂️ Working directory: {working_dir_characters} ")

        character_registry_path = os.path.join(working_dir_characters, "character_registry.json")
        if os.path.exists(character_registry_path):
            with open(character_registry_path, 'r', encoding='utf-8') as f:
                character_registry = json.load(f)
        else:
            character_registry = {}


        # Extract characters from script if not provided
        if len(character_registry) > 0:
            characters = []
            for identifier_in_scene in list(character_registry.keys()):
                with open(os.path.join(working_dir_characters, f"{identifier_in_scene}.json"), 'r', encoding='utf-8') as f:
                    character_data = json.load(f)
                    character = CharacterInScene.model_validate(character_data)
                    characters.append(character)
            print(f"⏭️ Skipped character extraction, loaded {len(characters)} characters from existing registry.")
        else:
            print(f"⬜ Extracting characters from script...")
            characters: list[CharacterInScene] = await self.character_extractor(script)
            for character in characters:
                with open(os.path.join(working_dir_characters, f"{character.identifier_in_scene}.json"), 'w', encoding='utf-8') as f:
                    json.dump(character.model_dump(), f, ensure_ascii=False, indent=4)
                character_registry[character.identifier_in_scene] = []
            print(f"☑️ Extracted {len(characters)} characters from script.")


        # Generate portraits for each character if not provided
        prompt_template = "Generate a full-body, front-view portrait of a character based on the following description, with an empty background. The character should be centered in the image, occupying most of the frame. Gazing straight ahead. Standing with arms relaxed at sides. Natural expression.\nfeatures: {features} \nstyle: {style}"

        async def generate_portrait_for_single_character(sem, character: CharacterInScene):
            if len(character_registry[character.identifier_in_scene]) == 0:
                async with sem:
                    features = "(static)" + character.static_features + ", (dynamic)" + character.dynamic_features
                    prompt = prompt_template.format(features=features, style=style)
                    image: ImageGeneratorOutput = await self.image_generator.generate_single_image(
                        prompt=prompt,
                        size="512x512",
                    )
                    portrait_path = os.path.join(working_dir_characters, f"{character.identifier_in_scene}.png")
                    image.save(portrait_path)
                    return character, portrait_path, "A front-view portrait of " + character.identifier_in_scene + "."
            else:
                print(f"⏭️ Skipped portrait generation for {character.identifier_in_scene}, already exists in registry.")
                return character, None, None

        sem = asyncio.Semaphore(3)
        tasks = []
        for character in characters:
            tasks.append(generate_portrait_for_single_character(sem, character))

        if len(tasks) > 0:
            print(f"⬜ Generating portraits for characters...")
        else:
            print(f"⏭️ Skipped portrait generation, all characters already exist in registry.")

        for task in asyncio.as_completed(tasks):
            character, portrait_path, description = await task
            if portrait_path is not None:
                character_registry[character.identifier_in_scene].append({
                    "path": portrait_path,
                    "description": description,
                })
                with open(character_registry_path, 'w', encoding='utf-8') as f:
                    json.dump(character_registry, f, ensure_ascii=False, indent=4)
                print(f"✔️ Generated portrait for {character.identifier_in_scene}, saved to {portrait_path}, and updated registry.")

        print(f"☑️ Finished generating portraits for characters.")

        return character_registry


    async def _design_storyboard_and_generate_shots(
        self,
        script: str,
        character_registry: Dict[str, List[Dict[str, str]]],
    ):
        working_dir = os.path.join(self.working_dir, "shots")
        os.makedirs(working_dir, exist_ok=True)
        print(f"🗂️ Working directory: {working_dir} ")

        reference_image_paths_dir = os.path.join(working_dir, "reference_image_paths")
        os.makedirs(reference_image_paths_dir, exist_ok=True)
        frame_candidates_dir = os.path.join(working_dir, "frame_candidates")
        os.makedirs(frame_candidates_dir, exist_ok=True)

        characters_identifiers = list(character_registry.keys())


        video_futures = []
        executor = ThreadPoolExecutor(max_workers=1)


        available_image_path_and_text_pairs = []
        for character_identifier, portraits in character_registry.items():
            for portrait in portraits:
                available_image_path_and_text_pairs.append((portrait["path"], portrait["description"]))


        # Design storyboard and generate frames
        print(f"⬜ Designing storyboard and generating frames...")
        existing_shots = []
        while True:
            current_shot_idx = len(existing_shots)
            print(f"   🎬 Processing shot {current_shot_idx}...")

            # 1. Design next storyboard shot
            shot_description_path = os.path.join(working_dir, f"{current_shot_idx}.json")
            if os.path.exists(shot_description_path):
                with open(shot_description_path, 'r', encoding='utf-8') as f:
                    shot_data = json.load(f)
                shot_description = Shot.model_validate(shot_data)
                print(f"⏭️ Skipped designing shot {current_shot_idx}, loaded from existing file.")
            else:
                start_time_design_shot = time.time()
                shot_description: Shot = await self.storyboard_generator.get_next_shot_description(
                    script=script,
                    character_identifiers=characters_identifiers,
                    existing_shots=existing_shots,
                )
                with open(shot_description_path, 'w', encoding='utf-8') as f:
                    json.dump(shot_description.model_dump(), f, ensure_ascii=False, indent=4)
                end_time_design_shot = time.time()
                duration_design_shot = end_time_design_shot - start_time_design_shot
                print(f"☑️ Designed new shot {current_shot_idx} and saved to {shot_description_path} (took {duration_design_shot:.2f} seconds).")

            existing_shots.append(shot_description)

            # 2. generate first (and last) frame candidates for the shot
            for frame_type in ["first_frame", "last_frame"]:
                if not hasattr(shot_description, frame_type) or getattr(shot_description, frame_type) is None:
                    logging.info(f"Shot {current_shot_idx} does not require {frame_type}, skipping generation.")
                    continue

                best_save_path = os.path.join(working_dir, f"{current_shot_idx}_{frame_type}.png")
                if os.path.exists(best_save_path):
                    print(f"⏭️ Skipped generating {frame_type} for shot {current_shot_idx}, already exists.")
                    continue

                start_time_generate_frame = time.time()

                # 2.1 select reference image and generate guidance prompt
                ref_path = os.path.join(reference_image_paths_dir, f"{current_shot_idx}_{frame_type}_reference.json")
                if os.path.exists(ref_path):
                    with open(ref_path, 'r', encoding='utf-8') as f:
                        reference = json.load(f)
                    print(f"⏭️ Skipped selecting reference image for {frame_type} of shot {current_shot_idx}, loaded from existing file.")
                else:
                    start_time_select_reference = time.time()
                    reference = self.reference_image_selector(
                        frame_description=getattr(shot_description, frame_type),
                        available_image_path_and_text_pairs=available_image_path_and_text_pairs,
                    )
                    with open(ref_path, 'w', encoding='utf-8') as f:
                        json.dump(reference, f, ensure_ascii=False, indent=4)
                    end_time_select_reference = time.time()
                    print(f"☑️ Selected reference image for {frame_type} of shot {current_shot_idx} and saved to {ref_path} (took {end_time_select_reference - start_time_select_reference:.2f} seconds).")


                # 2.2 generate frame candidates
                num_candidates = 3
                cur_frame_candidates_dir = os.path.join(frame_candidates_dir, f"shot_{current_shot_idx}_{frame_type}")
                print(f"   🎨 Generating {frame_type} candidates for shot {current_shot_idx}...")
                os.makedirs(cur_frame_candidates_dir, exist_ok=True)
                existing_frames = os.listdir(cur_frame_candidates_dir)
                missing_indices = [i for i in range(num_candidates) if f"{i}.png" not in existing_frames]
                if len(existing_frames) >= num_candidates:
                    print(f"⏭️ Skipped generating {frame_type} for shot {current_shot_idx}, already have {len(existing_frames)} candidates.")
                else:
                    print(f"⬜ Generating {num_candidates - len(existing_frames)} candidates for {frame_type} of shot {current_shot_idx}...")
                    start_time_generate_candidates = time.time()
                    prompt = reference["text_prompt"]
                    reference_image_paths = [path for path, _ in reference["reference_image_path_and_text_pairs"]]
                    num_images = num_candidates - len(existing_frames)

                    tasks = []
                    for _ in range(num_images):
                        task = self.image_generator.generate_single_image(
                            prompt=prompt,
                            reference_image_paths=reference_image_paths,
                            size="1600x900",
                        )
                        tasks.append(task)
                    images: List[ImageGeneratorOutput] = await asyncio.gather(*tasks)
                    for idx, image in enumerate(images):
                        image_path = os.path.join(cur_frame_candidates_dir, f"{missing_indices[idx]}.png")
                        image.save(image_path)
                        print(f"✔️ Generated candidate {missing_indices[idx]} for {frame_type} of shot {current_shot_idx}, saved to {image_path}.")

                    # for idx, task in enumerate(asyncio.as_completed(tasks)):
                    #     image: ImageGeneratorOutput = await task
                    #     image_path = os.path.join(cur_frame_candidates_dir, f"{missing_indices[idx]}.png")
                    #     image.save(image_path)
                    #     print(f"✔️ Generated candidate {missing_indices[idx]} for {frame_type} of shot {current_shot_idx}, saved to {image_path}.")

                    end_time_generate_candidates = time.time()
                    print(f"☑️ Generated {num_images} candidates for {frame_type} of shot {current_shot_idx} (took {end_time_generate_candidates - start_time_generate_candidates:.2f} seconds).")

                # 2.3 select the best frame candidate
                print(f"🏆 Selecting best image from candidates...")
                start_time_select_best = time.time()
                ref_image_path_and_text_pairs = reference["reference_image_path_and_text_pairs"]
                target_description = getattr(shot_description, frame_type)
                candidate_image_paths = [os.path.join(cur_frame_candidates_dir, f) for f in os.listdir(cur_frame_candidates_dir)]
                best_image_path = await self.best_image_selector(
                    ref_image_path_and_text_pairs=ref_image_path_and_text_pairs,
                    target_description=target_description,
                    candidate_image_paths=candidate_image_paths,
                )
                shutil.copy(best_image_path, best_save_path)
                end_time_select_best = time.time()
                print(f"☑️ Selected best image for {frame_type} of shot {current_shot_idx} and saved to {best_save_path} (took {end_time_select_best - start_time_select_best:.2f} seconds).")

                end_time_generate_frame = time.time()
                duration_generate_frame = end_time_generate_frame - start_time_generate_frame
                print(f"☑️ Generated {frame_type} for shot {current_shot_idx} (took {duration_generate_frame:.2f} seconds).")


                available_image_path_and_text_pairs.append((best_save_path, target_description))


            #  Submit background video generation immediately after frames are ready

            video_path = os.path.join(working_dir, f"{shot_description.idx}_video.mp4")
            if os.path.exists(video_path):
                print(f"⏭️ Skipped generating video for shot {shot_description.idx}, already exists.")
            else:
                print(f" 🚀 Submitting background video generation for shot {shot_description.idx}...")
                frame_paths = []
                if hasattr(shot_description, "first_frame") and shot_description.first_frame:
                    first_frame_path = os.path.join(working_dir, f"{shot_description.idx}_first_frame.png")
                    frame_paths.append(first_frame_path)

                if hasattr(shot_description, "last_frame") and shot_description.last_frame:
                    last_frame_path = os.path.join(working_dir, f"{shot_description.idx}_last_frame.png")
                    frame_paths.append(last_frame_path)
                future = executor.submit(
                    self._run_video_with_retries,
                    shot_description.visual_content,
                    frame_paths,
                    video_path,
                    3, # max_attempts=3,
                    5, # delay seconds
                )
                ensure_start_deadline = time.time() + 1.0
                while not future.running() and time.time() < ensure_start_deadline:
                    time.sleep(0.05)
                if future.running():
                    print(f"   ▶️ Video task is running for shot {shot_description.idx}")
                video_futures.append((shot_description.idx, future))

            if shot_description.is_last:
                break


        if video_futures:
            print(f"⏳ Waiting for {len(video_futures)} background video task(s) to complete...")
            wait_start = time.time()
            for shot_idx, future in video_futures:
                try:
                    future.result()
                    print(f"   ✅ Video task completed for shot {shot_idx}")
                except Exception as e:
                    logging.error(f"Video generation task failed for shot {shot_idx}: {e}")
                    print(f"   ❌ Video task failed for shot {shot_idx}: {str(e)}")
            wait_duration = time.time() - wait_start
            print(f"✅ All background video tasks completed in {wait_duration:.2f}s")
        else:
            print("📁 All videos already exist, skipping generation")
        executor.shutdown(wait=True)

    def _run_video_with_retries(
        self,
        prompt: str,
        frame_paths: list,
        save_path: str,
        max_attempts: int = 3,
        delay_seconds: float = 5.0,
    ) -> str:
        """Run video generation with retries. Returns save_path on success, raises on final failure."""
        last_error = None
        for attempt in range(1, max_attempts + 1):
            try:
                logging.info(f"[VideoRetry] Attempt {attempt}/{max_attempts} for {save_path}")
                video = asyncio.run(self.video_generator.generate_single_video(prompt, frame_paths))
                video.save(save_path)
                if os.path.exists(save_path):
                    logging.info(f"[VideoRetry] Success on attempt {attempt} for {save_path}")
                    return save_path
                else:
                    logging.warning(f"[VideoRetry] No output file after attempt {attempt} for {save_path}")
            except Exception as e:
                last_error = e
                logging.error(f"[VideoRetry] Exception on attempt {attempt} for {save_path}: {e}")
            if attempt < max_attempts:
                time.sleep(delay_seconds)
        # If we get here, all attempts failed
        error_message = f"Video generation failed after {max_attempts} attempts for {save_path}"
        if last_error:
            raise RuntimeError(error_message) from last_error
        raise RuntimeError(error_message)
    


        # print(f"✅ Designed storyboard and generated all frames.")
        # self.video_generator: BaseVideoGenerator

        # shots = existing_shots
        # # Generate video for the shot
        # async def generate_video_for_single_shot(sem, shot: Shot):
        #     async with sem:
        #         video_path = os.path.join(working_dir, f"{shot.idx}_video.mp4")
        #         if os.path.exists(video_path):
        #             print(f"⏭️ Skipped generating video for shot {shot.idx}, already exists.")
        #             return

        #         reference_image_paths = []

        #         if hasattr(shot, "first_frame") and shot.first_frame:
        #             first_frame_path = os.path.join(working_dir, f"{shot.idx}_first_frame.png")
        #             reference_image_paths.append(first_frame_path)
        #         if hasattr(shot, "last_frame") and shot.last_frame:
        #             last_frame_path = os.path.join(working_dir, f"{shot.idx}_last_frame.png")
        #             reference_image_paths.append(last_frame_path)

        #         prompt = shot.visual_content
        #         video: VideoGeneratorOutput = await self.video_generator.generate_single_video(
        #             prompt=prompt,
        #             reference_image_paths=reference_image_paths,
        #         )
        #         video.save(video_path)
        #         print(f"☑️ Generated video for shot {shot.index} and saved to {video_path}.")


        # # Generate video for the shot
        # sem = asyncio.Semaphore(3)
        # tasks = []
        # for shot in shots:
        #     tasks.append(generate_video_for_single_shot(sem, shot))

        # await asyncio.gather(*tasks)
        # print(f"✅ Finished generating videos for all shots.")
