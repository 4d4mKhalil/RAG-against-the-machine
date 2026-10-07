import json,os
from path import Path
from .aug_generator import augmenter
from .models import Chunk, MinimalAnswer, MinimalSearchResults, RagDataset, StudentSearchResults, StudentSearchResultsAndAnswer, UnansweredQuestion
from rich import print
from typing import List
from tqdm import tqdm
from .indexer import indexer
from .retriever import chunks_loader, searcher

class Pipeline:
    def index(self, max_chunk_size: int = 2000) -> None:
        processed = Path("./data/processed")
        if not ((processed / "bm25_index").exists()
            and (processed / "chunks.json").exists()
            and (processed / "faiss_index").exists()
        ):
            indexer('./data/raw', './data/processed', max_chunk_size)
        print(f"Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 10) -> None:
        processed = Path("./data/processed")
        if not ((processed / "bm25_index").exists()
            and (processed / "chunks.json").exists()
            and (processed / "faiss_index").exists()
        ):
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
        processed = Path("./data/processed")
        if not ((processed / "bm25_index").exists()
            and (processed / "chunks.json").exists()
            and (processed / "faiss_index").exists()
        ):
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
        output_filename = Path(dataset_path).name
        with open(f"{save_directory}/{output_filename}", "w") as f:
            f.write(studen_search_res.model_dump_json(indent=4))

    def answer(self, query: str, k: int = 5) -> None:
        processed = Path("./data/processed")

        if not ((processed / "bm25_index").exists()
            and (processed / "chunks.json").exists()
            and (processed / "faiss_index").exists()
        ):
            indexer('./data/raw', './data/processed', 2000)
        top_k_chunks = searcher(query, k, './data/processed')
        answer = augmenter(query, top_k_chunks)
        print('\n', answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer"
    ) -> None:
        processed = Path("./data/processed")

        if not ((processed / "bm25_index").exists()
            and (processed / "chunks.json").exists()
            and (processed / "faiss_index").exists()
        ):
            indexer('./data/raw', './data/processed', 2000)
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
        

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        print(f"evaluate called")