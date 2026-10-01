class Pipeline:
    def index(self, max_chunk_size: int = 2000) -> None:
        print(f"index called, max_chunk_size={max_chunk_size}")

    def search(self, query: str, k: int = 10) -> None:
        print(f"search called: {query!r} k={k}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results"
    ) -> None:
        print(f"search_dataset called")

    def answer(self, query: str, k: int = 5) -> None:
        print(f"answer called: {query!r} k={k}")

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