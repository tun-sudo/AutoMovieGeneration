from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple



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
        description="Describe the dynamic changes within the shot (camera movement and the movement of elements within the frame). Here are some examples: 'The animation ignites with a Rosso Corsa Ferrari supercar model initially as a dynamic blur of motion against a sleek, dark gradient or minimalist studio background. Within 1 second, it snaps into sharp focus, revealing its aggressive front fascia and iconic Prancing Horse emblem. (Seconds 1-3): A quick, cinematic camera sweep glides from the emblem, across a signature sculpted hood line or a low-profile aerodynamic side mirror, then tightly frames a detailed multi-spoke alloy wheel with visible brake calipers. (Seconds 3-4): A brief, impactful shot showcasing the powerful rear haunches and integrated spoiler, perhaps with a glint of light off the taillights.' ; 'Static camera, extreme close-up on the woman's face. Her deep sapphire-blue eyes, flecked with gold, dominate the frame, framed by subtle crow's feet and faint freckles on her pale, porcelain skin. The atmosphere evokes quiet heartbreak, with high detail in skin texture (fine pores and soft wrinkles), saturated colors in tear reflections (vibrant blues and golds), natural light simulation via gentle diffusion from an overcast evening sky, and dynamic micro-motion in the tears' quiver and fall, plus a faint emotional tremor in her gaze. .' ; ' Lego-style racecar bursting through a brick pit stop wall, pieces flying, plants and grass around the base, brown ground and pavement background, dust kicking up, dynamic motion. High detail, saturated Lego colors, HDR contrast, sharp shadows, natural light simulation' ; etc. "
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



