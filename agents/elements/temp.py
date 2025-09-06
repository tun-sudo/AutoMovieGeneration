# from agents.novel_compressor import NovelCompressor
# from agents.event_extractor import EventExtractor
# from agents.scene_extractor import SceneExtractor
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.output_parsers import JsonOutputParser




class Character(BaseModel):
    index: int = Field(
        description="Unique index of the character",
        examples=[0, 1, 2],
    )
    identifier: str = Field(
        description="Unique identifier for the character. For important character, usually their full name. For background characters, usually their identity, title, or characteristics.",
        examples=["John Doe", "Doctor Wang", "A Mysterious Stranger"]
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="List of aliases for the character, such as nicknames and titles",
        examples=["Johnny", "The Doc", "Stranger"]
    )
    character_traits: Optional[str] = Field(
        default=None,
        description="Description of the character's traits that do not change throughout the story. For a story with a relatively short time span or a cultivation novel where appearances remain unchanged, the unchanging characteristic of a character is often his/her face and figure. For a story with a longer time span, such as a person's entire life, there may be no unchanging physical features, and this field should be left empty.",
        examples=["Tall and slender with sharp blue eyes", "Short and stout with a bushy beard", None]
    )



class Environment(BaseModel):
    index: int = Field(
        description="Unique index of the scene setting",
        examples=[0, 1, 2],
    )
    setting: str = Field(
        description="Description of the scene setting, including INT. or EXT., location, time and other details",
        examples=[
            "INT. Home Office - DAY\nA dusty old study at dusk. Thick, honeyed light pours through the west-facing window, cutting through the gloom and illuminating countless drifting dust motes.\nA massive desk sits flush against the inner wall. Upon it, a lamp, stacks of old books, a pair of glasses, and an open leather-bound notebook each hold their place in the slanted light. A globe stands quietly to the side.\nAcross the room, floor-to-ceiling bookshelves cast long, uneven shadows. The fireplace is cold; picture frames on the mantel gleam dully in the dark. A worn leather armchair sinks into a corner, a book left face-down on the seat.\nThe light feels still, the silence deep, as though time itself has been sealed away here."
        ]
    )

    class EnvironmentInSpecificScene(BaseModel):
        scene_index: int = Field(
            description="Index of the scene where this setting was used",
            examples=[0, 1, 2],
        )
        atmosphere: Optional[str] = Field(
            default=None,
            description="Description of any changes to the setting that occur in this scene.",
            examples=["The room is now brightly lit", "The weather has turned stormy", None]
        )

    scene_records: List[EnvironmentInSpecificScene] = Field(
        default_factory=list,
        description="The list of scenes where this environment was used",
    )


class Scene(BaseModel):
    index: int = Field(
        description="Unique index of the scene"
    )
    environment_index: int = Field(
        description="Index of the environment used in this scene"
    )
    character_indices: List[int] = Field(
        default_factory=list,
        description="List of character indices present in this scene"
    )

    class CharacterInSpecificScene(BaseModel):
        scene_index: int = Field(
            description="Index of the scene in which the character appears",
            examples=[0, 1, 2],
        )
        character_states: str = Field(
            description="The field is distinguished from traits. It describes the character's appearance and traits that may change from scene to scene. It typically describes the character's clothing in the scene or other changing appearance characteristics.",
            examples=["Wearing a tattered cloak", "With a new haircut", "Looking older and more worn"]
        )
        actions: str = Field(
            description="What the character does in this scene",
            examples=["Draws his sword and charges into battle", "Sits quietly reading a book", "Argues passionately with another character"]
        )

    scene_records: List[CharacterInSpecificScene] = Field(
        default_factory=list,
        description="The list of scenes where the character appears",
    )

    script: str = Field(
        description="Script for the scene"
    )


class Event(BaseModel):
    index: int = Field(
        description="Unique index of the event",
    )
    scene_indices: List[int] = Field(
        default_factory=list,
        description="List of scene indices associated with this event"
    )


parser = JsonOutputParser(pydantic_object=Character)
print(parser.get_format_instructions())