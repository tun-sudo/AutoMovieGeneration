import json
import os
from agents import *
import logging




class VideoGenerationPipeline:
    def __init__(
        self,
        base_url,
        api_key,
        working_dir: str = ".working_dir",
    ):
        os.makedirs(working_dir, exist_ok=True)
        self.working_dir = working_dir

        self.storyboard_generator = StoryboardGenerator(
            model="gpt-5-2025-08-07",
            base_url=base_url,
            api_key=api_key
        )

        self.character_generator = CharacterImageGenerator(
            base_url=base_url,
            api_key=api_key
        )

        self.reference_image_selector = ReferenceImageSelector(
            model="gpt-5-2025-08-07",
            base_url=base_url,
            api_key=api_key
        )

        self.frame_image_generator = FrameImageGenerator(
            base_url=base_url,
            api_key=api_key,
            chat_model="gpt-5-2025-08-07",
        )

        self.image_consistency_checker = ImageConsistencyChecker(
            model="gpt-5-2025-08-07",
            base_url=base_url,
            api_key=api_key
        )

        self.video_generator = VideoGenerator(
            base_url=base_url,
            api_key=api_key,
        )

    def __call__(
        self,
        script: str,
        style: str,
    ):
        available_image_path_and_text_pairs = []

        # 1. extract all characters in the script (global information)
        logging.info(f"Extracting characters from script")
        character_text_dir = os.path.join(self.working_dir, "text", "characters")
    
        # extract and save the characters
        characters = self.storyboard_generator.extract_characters(script)
        os.makedirs(character_text_dir, exist_ok=True)
        for character in characters:
            with open(f"{character_text_dir}/character_{character.idx}.json", "w") as f:
                json.dump(character.model_dump(), f, indent=4, ensure_ascii=False)

        # load the characters
        characters = []
        for file_name in os.listdir(character_text_dir):
            if file_name.endswith(".json"):
                with open(os.path.join(character_text_dir, file_name), "r") as f:
                    character_dict = json.load(f)
                    character: Character = Character.model_validate(character_dict)
                    characters.append(character)

        # 2. generate the character portrait
        character_image_dir = os.path.join(self.working_dir, "images", "characters")
        os.makedirs(character_image_dir, exist_ok=True)
        for character in characters:
            # generate and save
            save_dir = os.path.join(character_image_dir, character.identifier)
            logging.info(f"Generating portrait for character: {character.identifier} (save to {save_dir})")
            os.makedirs(save_dir, exist_ok=True)
            self.character_generator(character, style, save_dir)

            for image_name in os.listdir(save_dir):
                view_image_path = os.path.join(save_dir, image_name)
                view = image_name.split(".")[0]  # front, side, back
                view_text = f"A {view}-view portrait of {character.identifier}"
                available_image_path_and_text_pairs.append((view_image_path, view_text))

        # 3. loop: design next shot -> generate first (and last) frame -> generate video
        shots = []
        shot_text_dir = os.path.join(self.working_dir, "text", "shots")
        os.makedirs(shot_text_dir, exist_ok=True)
        shot_image_dir = os.path.join(self.working_dir, "images", "frames")
        os.makedirs(shot_image_dir, exist_ok=True)
        while True:
            # 3.1 design next shot
            shot = self.storyboard_generator.get_next_shot_description(script, characters, shots)

            # save the shot
            with open(f"{shot_text_dir}/shot_{shot.idx}.json", "w") as f:
                json.dump(shot.model_dump(), f, indent=4, ensure_ascii=False)

            # load the shot
            with open(f"{shot_text_dir}/shot_{len(shots)}.json", "r") as f:
                shot_dict = json.load(f)
                shot: Shot = Shot.model_validate(shot_dict)

            shots.append(shot)


            frame_paths = []
            # 3.2 generate the first frame (and last frame) of the shot
            for frame_type in ["first_frame", "last_frame"]:
                if not getattr(shot, frame_type):
                    logging.info(f"Shot {shot.idx} does not have {frame_type} attribute")
                    continue

                # 3.2.1 select refence image and generate guidance prompt
                logging.info(f"Selecting reference images for shot {shot.idx} - {frame_type}")
                ref_image_indices_and_text_prompt = self.reference_image_selector(available_image_path_and_text_pairs, getattr(shot, frame_type))
                ref_image_path_and_text_pairs = [
                    available_image_path_and_text_pairs[i]
                    for i in ref_image_indices_and_text_prompt.ref_image_indices
                ]
                guide_prompt = ref_image_indices_and_text_prompt.text_prompt

                # generate candidate images, then select the best one
                logging.info(f"Reference images included for shot {shot.idx} - {frame_type}: \n{ref_image_path_and_text_pairs}")
                logging.info(f"Generating {frame_type} for shot {shot.idx}: \n{guide_prompt}")
                candidate_save_dir = os.path.join(shot_image_dir, f"shot_{shot.idx}-{frame_type}_candidate")
                os.makedirs(candidate_save_dir, exist_ok=True)
                best_save_path = os.path.join(shot_image_dir, f"shot_{shot.idx}-{frame_type}.png")

                self.frame_image_generator(
                    ref_image_path_and_text_pairs=ref_image_path_and_text_pairs,
                    guide_prompt=guide_prompt,
                    frame_description=getattr(shot, frame_type),
                    candidate_save_dir=candidate_save_dir,
                    best_save_path=best_save_path,
                )

                # while True:
                #     # 3.2.2 generate the frame image
                #     logging.info(f"Generating {frame_type} for shot {shot.idx}: \n{guide_prompt}")
                #     self.frame_image_generator(
                #         ref_image_path_and_text_pairs,
                #         guide_prompt,
                #         save_path,
                #     )

                #     # 3.2.3 check consistency
                #     logging.info(f"Checking consistency for shot {shot.idx} - {frame_type}")
                #     consistency_result = self.image_consistency_checker(
                #         ref_image_path_and_text_pairs,
                #         guide_prompt,
                #         save_path,
                #     )
                #     if consistency_result.is_consistent:
                #         logging.info(f"Shot {shot.idx} - {frame_type} is consistent with the description.")
                #         break
                #     else:
                #         logging.info(f"Shot {shot.idx} - {frame_type} is NOT consistent with the description.")
                #         logging.info(f"Reason: {consistency_result.reason}")
                #         guide_prompt = consistency_result.rectified_guide_prompt


                frame_paths.append(best_save_path)
                available_image_path_and_text_pairs.append((best_save_path, getattr(shot, frame_type)))


            # 4. generate video
            os.makedirs(os.path.join(self.working_dir, "videos"), exist_ok=True)
            video_save_path = os.path.join(self.working_dir, "videos", f"shot_{shot.idx}.mp4")
            self.video_generator.generate_video(
                prompt=shot.visual_content,
                image_paths=frame_paths,
                save_path=video_save_path,
            )

            if shot.is_last:
                break

        return shots
