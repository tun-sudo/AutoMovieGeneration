from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple



class Character(BaseModel):
    idx: int = Field(
        description="The index of the character in the script, starting from 0."
    )
    identifier: str = Field(
        description="The core unique name of the character, typically the most formal full name."
    )
    features: str = Field(
        description="The appearance details of the character, including face, hair, clothes, and other features. For example, 'big eyes, long hair, red dress, medium build'."
    )

    # face: str = Field(
    #     description="The face feature of the character. For example, 'big eyes', 'oval face', 'oriental', 'dimple', etc."
    # )
    # hair: str = Field(
    #     description="The hair style of the character. For example, 'short hair', 'long hair', 'curly hair', etc."
    # )
    # clothes: str = Field(
    #     description="The clothes of the character. For example, 'red shirt', 'blue jeans', 'black dress', etc."
    # )
    # other: Optional[str] = Field(
    #     None,
    #     description="Any other distinguishing features of the character except the face and hair. For example, 'mechanical arm', 'medium build', etc."
    # )


    def __str__(self):
        s = f"Character {self.idx}: {self.identifier}"
        s += f"\n  features: {self.features}"
        return s
    



class Shot(BaseModel):
    idx: int = Field(
        description="The index of the shot in the sequence, starting from 0."
    )
    is_last: bool = Field(
        description="Whether this is the last shot in the sequence. If True, no more shots will be planned after this one."
    )

    duration: str = Field(
        description="The estimated duration of the shot, typically between 3 to 10 seconds.",
    )

    # visual
    first_frame: str = Field(
        ...,
        description="Detail the static first frame of the shot (composition, characters, setting, lighting, color, etc.), emphasizing visual details. For example, 'A close-up shot of a woman's face positioned on the left side of the frame, with a window on the right, as she gazes toward the window.'"
    )
    visual_content: str = Field(
        ...,
        description="Describe the dynamic changes within the shot (camera movement and the movement of elements within the frame). For example, 'Static camera. Tears well up in the woman's eyes and eventually fall out of the frame.'"
    )
    last_frame: Optional[str] = Field(
        default=None,
        description="When there is a significant difference between the last frame and the first frame, particularly if a new character appears in the last frame, it is necessary to provide a description of the last frame. Otherwise, this field can be left empty."
    )


    # audio
    sound_effect: Optional[str] = Field(
        default=None,
        description="The sound effects used in the shot. For example, a door creaking or footsteps approaching."
    )
    speaker: Optional[str] = Field(
        default=None,
        description="The speaker in the shot, if applicable. Not more than one. The speaker identifier must match one of the character identifiers."
    )
    line: Optional[str] = Field(
        default=None,
        description="The dialogue or monologue in the shot, if applicable."
    )


    def __str__(self):
        s = f"Shot {self.idx}:"
        s += f"\n\tDuration: {self.duration}"
        # s += f"\n\tCamera Type: {self.camera_type}"
        # s += f"\n\tCamera Angle: {self.camera_angle}"
        # s += f"\n\tCamera Movement: {self.camera_movement}"
        # s += f"\n\tVisual Description: {self.visual_description}"
        # s += f"\n\tFirst Frame Visual Description: {self.first_frame_visual_description}"
        s += f"\n\tVisual Content: {self.visual_content}"
        # s += f"\n\tLast Frame Visual Description: {self.last_frame_visual_description}"
        s += f"\n\tSound Effects: {self.sound_effect}"
        s += f"\n\tSpeaker: {self.speaker}"
        s += f"\n\tLine: {self.line}"
        return s

    def __repr__(self):
        return self.__str__()



