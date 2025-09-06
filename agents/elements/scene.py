from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple




class Scene(BaseModel):
    idx: int = Field(
        ...,
        description="The index of the scene within the event, starting from 0"
    )
    slugline: str = Field(
        ...,
        description="The detailed description of the spatiotemporal information of the scene",
    )
    characters: List[str] = Field(
        ...,
        description="List of characters appearing in this scene, including main characters, supporting characters, and extras.",
    )
    script: str = Field(
        ...,
        description="The script of the scene, including actions and dialogues"
    )
