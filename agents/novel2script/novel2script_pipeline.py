import os
import logging
import json
import asyncio
from agents.novel2script.novel_compressor import NovelCompressor
from agents.novel2script.event_extractor import EventExtractor
from agents.novel2script.scene_extractor import SceneExtractor, Scene
from agents.elements.event import Event
from agents.novel2script.character_merger import CharacterMerger, CharacterAcrossScene, CharacterAcrossEvent

class Novel2ScriptPipeline:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        working_dir: str = ".cache",
    ):
        self.novel_compressor = NovelCompressor(
            chat_model="gpt-5-mini-2025-08-07",
            api_key=api_key,
            base_url=base_url,
        )
        self.event_extractor = EventExtractor(
            chat_model="gpt-5-mini-2025-08-07",
            api_key=api_key,
            base_url=base_url,
        )
        self.scene_extractor = SceneExtractor(
            chat_model="gpt-5-2025-08-07",
            api_key=api_key,
            base_url=base_url,
            knowledge_base_cache_dir=os.path.join(working_dir, "knowledge_base_cache"),
        )
        self.character_merger = CharacterMerger(
            chat_model="gpt-5-mini-2025-08-07",
            api_key=api_key,
            base_url=base_url,
        )


        self.working_dir = working_dir
        os.makedirs(self.working_dir, exist_ok=True)


        self.cache = {}
        self.cache_path = os.path.join(self.working_dir, "cache.json")
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def save_to_cache(self, *keys, variable_name, value):
        cache = self.cache
        for key in keys:
            if key not in cache:
                cache[key] = {}
            cache = cache[key]
        cache[variable_name] = value
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)


    def check_if_in_cache(self, *keys, variable_name):
        cache = self.cache
        for key in keys:
            if key not in cache:
                return False
            cache = cache[key]
        return variable_name in cache


    def load_from_cache(self, *keys, variable_name):
        cache = self.cache
        for key in keys:
            cache = cache[key]
        return cache[variable_name]


    def __call__(
        self,
        novel_text: str,
        max_concurrent_tasks: int = 3,
    ):

        logging.info("Starting novel to script pipeline...")

        # 1. compress the novel text to reduce token consumption
        logging.info("Compressing novel...")

        # 1.1 split into chunks
        logging.info("Splitting novel into chunks...")
        novel_chunks = self.novel_compressor.split(novel_text)
        logging.info(f"Novel split into {len(novel_chunks)} chunks.")

        # 1.2 compress each chunk
        cache_keys = ["compress novel", "compress chunks"]
        logging.info("Compressing novel chunks...")
        all_compressed_chunks = [None] * len(novel_chunks)
        uncompressed_index_chunk_pairs = []
        for idx, novel_chunk in enumerate(novel_chunks):
            if self.check_if_in_cache(*cache_keys, variable_name=f"compressed_chunk_{idx}"):
                logging.info(f"Loading compressed chunk {idx} from cache...")
                compressed_chunk = self.load_from_cache(*cache_keys, variable_name=f"compressed_chunk_{idx}")
                all_compressed_chunks[idx] = compressed_chunk
            else:
                uncompressed_index_chunk_pairs.append((idx, novel_chunk))

        if uncompressed_index_chunk_pairs:
            compressed_results = asyncio.run(
                self.novel_compressor.compress(
                    uncompressed_index_chunk_pairs,
                    max_concurrent_tasks=max_concurrent_tasks,
                )
            )
            for idx, compressed_chunk in compressed_results:
                all_compressed_chunks[idx] = compressed_chunk
                self.save_to_cache(*cache_keys, variable_name=f"compressed_chunk_{idx}", value=compressed_chunk)

        logging.info("Novel chunks compressed.")

        # 1.3 aggregate compressed chunks
        cache_keys = ["compress novel", "aggregate compressed chunks"]
        logging.info("Aggregating compressed chunks...")
        if self.check_if_in_cache(*cache_keys, variable_name="aggregated_compressed_novel"):
            logging.info("Loading aggregated compressed novel from cache...")
            aggregated_compressed_novel = self.load_from_cache(*cache_keys, variable_name="aggregated_compressed_novel")
        else:
            if len(all_compressed_chunks) > 1:
                aggregated_compressed_novel = self.novel_compressor.aggregate(all_compressed_chunks)
            else:
                aggregated_compressed_novel = all_compressed_chunks[0]
            self.save_to_cache(*cache_keys, variable_name="aggregated_compressed_novel", value=aggregated_compressed_novel)
        logging.info("Aggregated compressed chunks.")

        compressed_novel = aggregated_compressed_novel
        with open(os.path.join(self.working_dir, "compressed_novel.txt"), "w", encoding="utf-8") as f:
            f.write(compressed_novel)
        logging.info("Novel compression completed.")



        # 2. extract events from the compressed novel
        logging.info("Extracting events from compressed novel...")
        cache_keys = ["extract events"]
        events = []
        if self.check_if_in_cache(*cache_keys, variable_name="events"):
            logging.info("Loading events from cache...")
            events = self.load_from_cache(*cache_keys, variable_name="events")
            events = [Event.model_validate(event) for event in events]

        while len(events) == 0 or not events[-1].is_last:
            logging.info(f"Extracting event {len(events)+1}...")
            event = self.event_extractor.extract_next_event(
                novel_text=compressed_novel,
                extracted_events=events,
            )
            events.append(event)
            self.save_to_cache(*cache_keys, variable_name="events", value=[e.model_dump() for e in events])

        with open(os.path.join(self.working_dir, "events.json"), "w", encoding="utf-8") as f:
            json.dump([e.model_dump() for e in events], f, ensure_ascii=False, indent=4)

        logging.info(f"Total {len(events)} events extracted.")



        # 3. extract scenes from each event
        logging.info("Extracting scenes from events...")
        cache_keys = ["extract scenes"]
        all_scenes = [[] for _ in range(len(events))]
        if self.check_if_in_cache(*cache_keys, variable_name="all_scenes"):
            logging.info("Loading scenes from cache...")
            all_scenes = self.load_from_cache(*cache_keys, variable_name="all_scenes")
            all_scenes = [[Scene.model_validate(scene) for scene in scenes] for scenes in all_scenes]

        unfinished_indices = []
        for event_idx, scenes in enumerate(all_scenes):
            if len(scenes) == 0 or not scenes[-1].is_last:
                event = events[event_idx]
                assert event_idx == event.idx
                unfinished_indices.append((event, scenes))
            else:
                logging.info(f"Scenes in event {event_idx+1} already extracted, skip.")

        if unfinished_indices:
            # build knowledge base from the compressed novel
            logging.info("Building knowledge base from compressed novel...")
            knowledge_base = self.scene_extractor.construct_knowledge_base(compressed_novel)
            logging.info("Knowledge base built.")
            # TODO: 优化这里的并发逻辑，可以同时让多个事件提取后续的场景，这里是简单的串行，逐个事件提取场景
            for event, previous_scenes in unfinished_indices:
                logging.info(f"Extracting scenes for event {event.idx+1}...")
                scenes = previous_scenes
                while True:
                    scene = self.scene_extractor.get_next_scene(
                        knowledge_base=knowledge_base,
                        event=event,
                        previous_scenes=previous_scenes,
                    )
                    scenes.append(scene)
                    all_scenes[event.idx] = scenes
                    self.save_to_cache(*cache_keys, variable_name="all_scenes", value=[[s.model_dump() for s in scenes] for scenes in all_scenes])
                    logging.info(f"Extracted scene {len(scenes)} for event {event.idx+1}")
                    if scene.is_last:
                        break
                logging.info(f"All scenes for event {event.idx+1} extracted.")

        for idx, scenes in enumerate(all_scenes):
            os.makedirs(os.path.join(self.working_dir, "scenes"), exist_ok=True)
            with open(os.path.join(self.working_dir, "scenes", f"event_{idx+1}_scenes.json"), "w", encoding="utf-8") as f:
                json.dump([s.model_dump() for s in scenes], f, ensure_ascii=False, indent=4)

        logging.info("All scenes extracted from all events.")


        # 4. merge characters across scenes
        logging.info("Merging characters across scenes for each event...")
        cache_keys = ["merge characters across scenes"]
        characters_each_event = [[] for _ in range(len(all_scenes))]
        if self.check_if_in_cache(*cache_keys, variable_name="characters_each_event"):
            logging.info("Loading merged characters from cache...")
            characters_each_event = self.load_from_cache(*cache_keys, variable_name="characters_each_event")
            characters_each_event = [[CharacterAcrossScene.model_validate(character) for character in characters] for characters in characters_each_event]

        unfinished_indices = []
        for event_idx, characters in enumerate(characters_each_event):
            if len(characters) == 0:
                unfinished_indices.append(event_idx)
            else:
                logging.info(f"Characters in event {event_idx+1} already merged, skip.")

        if unfinished_indices:
            for event_idx in unfinished_indices:
                logging.info(f"Merging characters for event {event_idx+1}...")
                scenes = all_scenes[event_idx]
                characters = self.character_merger.merge_character_across_scenes(scenes)
                characters_each_event[event_idx] = characters
                self.save_to_cache(*cache_keys, variable_name="characters_each_event", value=[[c.model_dump() for c in characters] for characters in characters_each_event])
                logging.info(f"Merged {len(characters)} characters for event {event_idx+1}.")

        for idx, characters in enumerate(characters_each_event):
            with open(os.path.join(self.working_dir, "scenes", f"event_{idx+1}_characters.json"), "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in characters], f, ensure_ascii=False, indent=4)

        logging.info("All characters merged across scenes for all events.")


        # 5. merge characters across events
        logging.info("Merging characters across events...")
        cache_keys = ["merge characters across events"]
        characters = []
        if self.check_if_in_cache(*cache_keys, variable_name="characters"):
            logging.info("Loading merged characters across events from cache...")
            characters = self.load_from_cache(*cache_keys, variable_name="characters")
            characters = [CharacterAcrossEvent.model_validate(character) for character in characters]
        else:
            characters = self.character_merger.merge_character_across_events(
                events=events,
                characters_each_events=characters_each_event,
            )
            self.save_to_cache(*cache_keys, variable_name="characters", value=[c.model_dump() for c in characters])
            logging.info(f"Merged {len(characters)} characters across events.")

        with open(os.path.join(self.working_dir, "characters.json"), "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in characters], f, ensure_ascii=False, indent=4)
        logging.info("Novel to script pipeline completed.")


