from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple



class Character(BaseModel):
    idx: int = Field(
        description="The index of the character in the script, starting from 0."
    )
    identifier: str = Field(
        description="The core unique name of the character, typically the most formal full name, may also be a famous person/animation character"
    )
    features: str = Field(
        description="The appearance details of the character, including full name, gender, age, ethnicity, face, hair, clothes, and other features. For example, 'big eyes, long hair, red dress, medium build."
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
    
