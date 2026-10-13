"""All project settings live in one class."""

from pathlib import Path


class Settings:
    """Defaults and small helpers. Never instantiated; use the class."""
    RAW_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"
    SEARCH_OUTPUT_DIR: str = "data/output/search_results"
    ANSWER_OUTPUT_DIR: str = "data/output/search_results_and_answer"

    BM25_NAME: str = "bm25_index"
    FAISS_NAME: str = "faiss_index"
    CHUNKS_NAME: str = "chunks.json"

    MAX_CHUNK_SIZE: int = 2000
    DEFAULT_CHUNK_SIZE: int = 2000
    EXTENSIONS: tuple[str, ...] = (
        ".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml"
    )
    IGNORED_DIRS: tuple[str, ...] = (
        ".git", "__pycache__", ".mypy_cache", ".venv"
    )

    DEFAULT_K: int = 5
    RRF_K: int = 60
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64

    LLM_MODEL: str = "Qwen/Qwen3-0.6B"
    MAX_NEW_TOKENS: int = 256

    @staticmethod
    def index_files(directory: str) -> list[Path]:
        """Return the paths that make up a complete index."""
        base = Path(directory)
        return [
            base / Settings.BM25_NAME,
            base / Settings.FAISS_NAME,
            base / Settings.CHUNKS_NAME,
        ]

    @staticmethod
    def missing_index_files(directory: str) -> list[Path]:
        """Return the index parts that do not exist yet."""
        return [p for p in Settings.index_files(directory) if not p.exists()]

    @staticmethod
    def index_exists(directory: str) -> bool:
        """Return True when the whole index is present."""
        return not Settings.missing_index_files(directory)

    @staticmethod
    def is_indexable(path: Path) -> bool:
        """Return True if the file extension is supported."""
        return path.suffix.lower() in Settings.EXTENSIONS

    @staticmethod
    def is_ignored(path: Path) -> bool:
        """Return True if any part of the path is an ignored dir."""
        return any(part in Settings.IGNORED_DIRS for part in path.parts)

    @staticmethod
    def validate_chunk_size(size: int) -> int:
        """Return a valid chunk size or raise ValueError."""
        if not isinstance(size, int) or size <= 0:
            raise ValueError("max_chunk_size must be a positive integer")
        if size > Settings.MAX_CHUNK_SIZE:
            raise ValueError(
                f"max_chunk_size must be <= {Settings.MAX_CHUNK_SIZE}"
            )
        return size

    @staticmethod
    def validate_k(k: int) -> int:
        """Return a valid k or raise ValueError."""
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        return k