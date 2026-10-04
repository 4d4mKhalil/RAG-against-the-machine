import json
from path import Path
from .aug_generator import augmenter
from .models import MinimalAnswer, MinimalSearchResults, RagDataset, StudentSearchResults, UnansweredQuestion
from rich import print
from typing import List
from .indexer import indexer
from .retriever import chunks_loader, searcher

class Pipeline:
    def index(self, max_chunk_size: int = 2000) -> None:
        indexer('./data/raw', './data/processed', max_chunk_size)
        print(f"Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        indexer('./data/raw', './data/processed', 2000)
        chunks = searcher(query, k, './data/processed')
        for chunk in chunks:
            print(f"{chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = "data/output/search_results"
        ) -> None:
        indexer('./data/raw', './data/processed', 2000)
        with open(dataset_path) as file:
            dataset = json.load(file)
        rag_questions = RagDataset.model_validate(dataset)
        
        search_res: list[MinimalSearchResults] = []
        for q in rag_questions.rag_questions:
            search_res.append(
                MinimalSearchResults(
                    question_id=q.question_id,
                    question=q.question,
                    retrieved_sources=searcher(q.question, k, './data/processed')
                    ))
        studen_search_res = StudentSearchResults(search_results=search_res, k=k)
        with open(f"{save_directory}/dataset_docs_public.json", "w") as f:
            f.write(studen_search_res.model_dump_json(indent=4))

    def answer(self, query: str, k: int = 5) -> None:
        processed = Path("./data/processed")

        if not ((processed / "bm25_index").exists()
            and (processed / "chunks.json").exists()
            and (processed / "faiss_index").exists()
        ):
            indexer('./data/raw', './data/processed', 2000)
        chunks = searcher(query, k, './data/processed')
        context = [src.text for src in chunks]
        answer = augmenter(query, context)
        print('\n', answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        print(f"answer_dataset called")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        print(f"evaluate called")