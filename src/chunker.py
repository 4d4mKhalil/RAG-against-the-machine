from dataclasses import dataclass


@dataclass
class Chunk:
    file_path: str
    text: str
    start_char: int
    end_char: int


class PythonChunker:
    def __init__(self, max_chunk_size):
        self.max_chunk_size = max_chunk_size


class MarkdownChunker:
    def __init__(self, max_chunk_size, overlap):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap