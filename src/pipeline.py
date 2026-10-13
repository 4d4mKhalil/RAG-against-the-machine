import json
import os
from pathlib import Path

from .settings import Settings
from .aug_generator import augmenter
from .models import (
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from rich import print
from typing import List
from tqdm import tqdm
from .indexer import indexer
from .retriever import chunks_loader, searcher

class Pipeline:
    def index(self, max_chunk_size: int = Settings.DEFAULT_CHUNK_SIZE) -> None:
        processed = Settings.PROCESSED_DIR
        try:
            Settings.validate_chunk_size(max_chunk_size)
        except ValueError as e:
            print(e)
        if not Settings.index_exists(processed):
            indexer(Settings.RAW_DIR, processed, max_chunk_size)
        print(f"Ingestion complete! Indices saved under {processed}")

    def search(self, query: str, k: int = Settings.DEFAULT_K) -> None:
        processed = Settings.PROCESSED_DIR
        try:
            Settings.validate_k(k)
        except ValueError as e:
            print(e)
        if not Settings.index_exists(processed):
            indexer(Settings.RAW_DIR, processed, Settings.DEFAULT_CHUNK_SIZE)
        chunks = searcher(query, k, processed)
        for chunk in chunks:
            print(f"{chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = Settings.DEFAULT_K,
        save_directory: str = Settings.SEARCH_OUTPUT_DIR
    ) -> None:
        processed = Settings.PROCESSED_DIR
        try:
            Settings.validate_k(k)
        except ValueError as e:
            print(e)
        if not Settings.index_exists(processed):
            indexer(Settings.RAW_DIR, processed, Settings.DEFAULT_CHUNK_SIZE)
        with open(dataset_path) as file:
            dataset = json.load(file)
        rag_questions = RagDataset.model_validate(dataset)
        
        search_res: list[MinimalSearchResults] = []
        for q in tqdm(rag_questions.rag_questions, desc="Searching dataset"):
            search_res.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=searcher(q.question, k, processed)
                )
            )
        studen_search_res = StudentSearchResults(search_results=search_res, k=k)

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        output_filename = Path(dataset_path).name

        with open(save_directory / output_filename, "w") as f:
            f.write(studen_search_res.model_dump_json(indent=4))
        print(f'Saved student_search_results to {save_directory}/{output_filename}')

    def answer(self, query: str, k: int = Settings.DEFAULT_K) -> None:
        processed = Settings.PROCESSED_DIR
        try:
            Settings.validate_k(k)
        except ValueError as e:
            print(e)
        if not Settings.index_exists(processed):
            indexer(Settings.RAW_DIR, processed, Settings.DEFAULT_CHUNK_SIZE)
        top_k_chunks = searcher(query, k, processed)
        answer = augmenter(query, top_k_chunks)
        print('\n', answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = Settings.ANSWER_OUTPUT_DIR
    ) -> None:
        processed = Settings.PROCESSED_DIR

        if not Settings.index_exists(processed):
            indexer(Settings.RAW_DIR, processed, Settings.DEFAULT_CHUNK_SIZE)
        with open(student_search_results_path) as file:
            ssr_data = json.load(file)
        ssr = StudentSearchResults.model_validate(ssr_data)

        
        lst_min_answers: list[MinimalAnswer] = []
        os.makedirs(save_directory, exist_ok=True)
        file_cache: dict[str, str] = {}
        for ss in tqdm(ssr.search_results):
            context: list[Chunk] = []

            for src in ss.retrieved_sources:
                if src.file_path not in file_cache:
                    try:
                        with open(src.file_path) as f:
                            file_cache[src.file_path] = f.read()
                    except OSError:
                        continue  
                text = file_cache[src.file_path]
                context.append(
                    Chunk(
                        **src.model_dump(),
                        text=text[
                            src.first_character_index:src.last_character_index
                        ],
                    )
                )

            lst_min_answers.append(
                MinimalAnswer(
                    question_id=ss.question_id,
                    question=ss.question,
                    retrieved_sources=ss.retrieved_sources,
                    answer=augmenter(ss.question, context),
                )
            )
            out = StudentSearchResultsAndAnswer(search_results=lst_min_answers, k=ssr.k)
            with open(f"{save_directory}/{Path(student_search_results_path).name}", "w") as f:
                f.write(out.model_dump_json(indent=4))
        

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> None:
        with open(student_search_results_path) as f:
            student = json.load(f)["search_results"]
        with open(dataset_path) as f:
            truth = {q["question_id"]: q["sources"] for q in json.load(f)["rag_questions"]}

        def hit(a, b):
            if a["file_path"] != b["file_path"]:
                return False
            inter = min(a["last_character_index"], b["last_character_index"]) \
                - max(a["first_character_index"], b["first_character_index"])
            if inter <= 0:
                return False
            la = a["last_character_index"] - a["first_character_index"]
            lb = b["last_character_index"] - b["first_character_index"]
            return inter / (la + lb - inter) >= 0.05

        print('Evaluation Results')
        for k in (1, 3, 5, 10):
            scores = []
            for r in student:
                gt = truth[r["question_id"]]
                top = r["retrieved_sources"][:k]
                scores.append(sum(any(hit(t, s) for t in top) for s in gt) / len(gt))
            print(f"Recall@{k}: {sum(scores) / len(scores):.3f}")