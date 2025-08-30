import os
import shutil
import json
import logging
from agents import *
from agents.elements import Character, Shot


class VideoGenerationPipeline:
    def __init__(
        self,
        config: dict,
        working_dir: str = ".working_dir",
    ):
        os.makedirs(working_dir, exist_ok=True)
        self.working_dir = working_dir

        self.storyboard_generator = StoryboardGenerator(**config["StoryboardGenerator"])

        self.character_generator = CharacterImageGenerator(**config["CharacterImageGenerator"])

        self.reference_image_selector = ReferenceImageSelector(**config["ReferenceImageSelector"])

        self.frame_candidate_images_generator = FrameCandidateImagesGenerator(**config["FrameCandidateImagesGenerator"])

        self.best_image_selector = BestImageSelector(**config["BestImageSelector"])

        self.video_generator = VideoGenerator(**config["VideoGenerator"])


    def __call__(
        self,
        script: str,
        style: str,
    ):
        available_image_path_and_text_pairs = []

        # 1. extract all characters in the script (global information)
        logging.info(f"Extracting characters from script")
        character_text_dir = os.path.join(self.working_dir, "text", "characters")
    
        if os.path.exists(character_text_dir) and len(os.listdir(character_text_dir)) > 0:
            # load the characters
            characters = []
            for file_name in os.listdir(character_text_dir):
                if file_name.endswith(".json"):
                    with open(os.path.join(character_text_dir, file_name), "r") as f:
                        character_dict = json.load(f)
                        character: Character = Character.model_validate(character_dict)
                        characters.append(character)
        else:
            # extract and save the characters
            characters = self.storyboard_generator.extract_characters(script)
            os.makedirs(character_text_dir, exist_ok=True)
            for character in characters:
                with open(f"{character_text_dir}/character_{character.idx}.json", "w") as f:
                    json.dump(character.model_dump(), f, indent=4, ensure_ascii=False)



        # 2. generate the character portrait
        character_image_dir = os.path.join(self.working_dir, "images", "characters")
        os.makedirs(character_image_dir, exist_ok=True)
        for character in characters:
            # generate and save
            save_dir = os.path.join(character_image_dir, character.identifier)
            os.makedirs(save_dir, exist_ok=True)
            if not os.path.exists(save_dir) or len(os.listdir(save_dir)) == 0:
                character_image_paths = self.character_generator(character, style, save_dir)
            else:
                character_image_paths = [
                    os.path.join(save_dir, image_name)
                    for image_name in os.listdir(save_dir)
                ]

            for character_image_path in character_image_paths:
                view = os.path.basename(character_image_path).split(".")[0]  # e.g., front
                view_text = f"A {view}-view portrait of {character.identifier}"
                available_image_path_and_text_pairs.append((character_image_path, view_text))

        # 3. loop: design next shot -> generate first (and last) frame -> generate video
        shots = []
        shot_text_dir = os.path.join(self.working_dir, "text", "shots")
        os.makedirs(shot_text_dir, exist_ok=True)
        shot_image_dir = os.path.join(self.working_dir, "images", "frames")
        os.makedirs(shot_image_dir, exist_ok=True)
        while True:
            # 3.1 design next shot
            shot_text_path = f"{shot_text_dir}/shot_{len(shots)}.json"
            if not os.path.exists(shot_text_path):
                logging.info(f"Designing shot {len(shots)}")
                shot = self.storyboard_generator.get_next_shot_description(script, characters, shots)
                logging.info(f"Shot {shot.idx} designed: \n{shot}")

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

                best_save_path = os.path.join(shot_image_dir, f"shot_{shot.idx}-{frame_type}.png")
                if not os.path.exists(best_save_path):
                    # 3.2.1 select reference image and generate guidance prompt
                    logging.info(f"Selecting reference images for shot {shot.idx} - {frame_type}")
                    ref_image_indices_and_text_prompt = self.reference_image_selector(available_image_path_and_text_pairs, getattr(shot, frame_type))
                    ref_image_path_and_text_pairs = [
                        available_image_path_and_text_pairs[i]
                        for i in ref_image_indices_and_text_prompt.ref_image_indices
                    ]
                    guide_prompt = ref_image_indices_and_text_prompt.text_prompt
                    logging.info(f"Reference images selected for shot {shot.idx} - {frame_type}: \n{ref_image_path_and_text_pairs}")

                    # generate candidate images, then select the best one
                    logging.info(f"Generating {frame_type} for shot {shot.idx}: \n{guide_prompt}")
                    candidate_save_dir = os.path.join(shot_image_dir, f"shot_{shot.idx}-{frame_type}_candidate")
                    os.makedirs(candidate_save_dir, exist_ok=True)

                    candidate_image_paths = self.frame_candidate_images_generator(
                        ref_image_path_and_text_pairs=ref_image_path_and_text_pairs,
                        guide_prompt=guide_prompt,
                        save_dir=candidate_save_dir,
                        num_images=3,
                    )

                    best_image_path = self.best_image_selector(
                        ref_image_path_and_text_pairs=ref_image_path_and_text_pairs,
                        target_description=getattr(shot, frame_type),
                        candidate_image_paths=candidate_image_paths,
                    )

                    shutil.copy(best_image_path, best_save_path)

                frame_paths.append(best_save_path)
                available_image_path_and_text_pairs.append((best_save_path, getattr(shot, frame_type)))

                # TODO 这里只简单地选择了最后面8张图像，后面可以改成通过文本筛选
                if len(available_image_path_and_text_pairs) > 8:
                    available_image_path_and_text_pairs = available_image_path_and_text_pairs[-8:]


            # 4. generate video
            os.makedirs(os.path.join(self.working_dir, "videos"), exist_ok=True)
            video_save_path = os.path.join(self.working_dir, "videos", f"shot_{shot.idx}.mp4")
            if not os.path.exists(video_save_path):
                self.video_generator(
                    prompt=shot.visual_content,
                    image_paths=frame_paths,
                    save_path=video_save_path,
                )

            if shot.is_last:
                break

        return shots
