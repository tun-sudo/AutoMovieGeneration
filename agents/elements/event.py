from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple


class Event(BaseModel):
    idx: int = Field(
        ...,
        description="The index of the event, starting from 0"
    )

    is_last: bool = Field(
        ...,
        description="Indicates if this is the last event in the sequence"
    )

    description: str = Field(
        ...,
        description="The detailed description of the event, including timeframe, location, characters involved, and key actions."
    )


    def __str__(self):
        s = f"<Event {self.idx}>\n{self.description}"
        return s
