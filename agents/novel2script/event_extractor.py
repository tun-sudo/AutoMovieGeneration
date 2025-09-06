import os
import logging
import asyncio
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from agents.elements.event import Event
import json


system_prompt_template_extract_events = \
"""
You are a highly skilled Literary Analyst AI. Your expertise is in narrative structure, plot deconstruction, and thematic analysis. You meticulously read and interpret prose to break down a story into its fundamental sequential events.

**TASK**
Extract the next event from the provided novel, following the sequence of the story and building upon the partially extracted events.

**INPUT**
1. The full text of the novel, which is enclosed within <NOVEL_TEXT_START> and <NOVEL_TEXT_END> tags
2. A sequence of already-extracted events (in order), which is enclosed within <EXTRACTED_EVENTS_START> and <EXTRACTED_EVENTS_END> tags. The sequence may be empty.

Below is an example input:

<NOVEL_TEXT_START>
Once upon a time in a faraway land, there lived a young girl named Ella. She was kind and gentle, always helping those in need. One day, while exploring the woods near her home, she stumbled upon a hidden path that led to a magical garden.
...
<NOVEL_TEXT_END>

<EXTRACTED_EVENTS_START>
<Event 0>
Ella discovers a hidden path in the woods ...
<Event 1>
Ella follows the hidden path and finds a magical garden ...
...
<EXTRACTED_EVENTS_END>


**OUTPUT**
{format_instructions}

**GUIDELINES**
1. Focus on events that are critical to the plot, character development, or thematic depth.
2. Ensure the event is logically distinct from previous and subsequent events.
3. If the event spans multiple scenes, unify them under a single dramatic goal. For example, a chase sequence might begin in a city market, continue through back alleys, and conclude on a rooftop—all comprising a single event because they collectively achieve the dramatic purpose of "the protagonist evading capture."
4. Maintain objectivity: describe events based on the text without interpretation or judgment.
5. For the description of event, you should include specific details about the timeframe, location, characters involved, and key actions. 
Below is an example:
Timeframe: The following morning, after acquiring the information about the Temple.
Characters: Elara (protagonist) and Kaelen (her rival treasure hunter).
Cause: Both seek the same artifact and are determined to reach it first.
Process: The event begins with Elara hastily purchasing supplies in the port town (scene 1), where she spots Kaelen already hiring a crew, raising the stakes. It continues as she races to secure her own ship and captain, negotiating fiercely under time pressure (scene 2). The event culminates in a direct confrontation on the docks (scene 3), where Kaelen attempts to sabotage her vessel, leading to a brief but intense sword fight between the two rivals.
Outcome: Elara successfully defends her ship and sets sail, but the conflict solidifies a bitter personal rivalry with Kaelen, ensuring their race to the temple will be fraught with direct opposition and danger.
6. Every detail in your event description must be directly supported by the input novel. Do not add, assume, or invent any information.
7. The language of outputs in values should be same as the input text.
"""

human_prompt_template_extract_next_event = \
"""
<NOVEL_TEXT_START>
{novel_text}
<NOVEL_TEXT_END>

<EXTRACTED_EVENTS_START>
{extracted_events}
<EXTRACTED_EVENTS_END>
"""



class EventExtractor:
    def __init__(
        self,
        chat_model: str,
        api_key: str,
        base_url: str,
    ):
        self.chat_model = init_chat_model(
            model=chat_model,
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
        )
        self.parser = PydanticOutputParser(pydantic_object=Event)


    def __call__(
        self,
        novel_text: str,
    ):
        logging.info("Extracting events from novel...")

        events = []
        while True:
            event = self.extract_next_event(novel_text, events)

            events.append(event)
            logging.info(f"Extracted event: \n{event}")
            if event.is_last:
                break

        return events

    def extract_next_event(
        self,
        novel_text: str,
        extracted_events: List[Event]
    ) -> Event:
        
        extracted_events_str = "\n".join([str(e) for e in extracted_events])

        messages = [
            SystemMessage(
                content=system_prompt_template_extract_events.format(format_instructions=self.parser.get_format_instructions()),
            ),
            HumanMessage(
                content=human_prompt_template_extract_next_event.format(
                    novel_text=novel_text,
                    extracted_events=extracted_events_str,
                )
            )
        ]

        chain = self.chat_model | self.parser

        event: Event = chain.invoke(messages)

        return event



