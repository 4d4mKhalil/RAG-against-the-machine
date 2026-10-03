from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    file_path: str
    text: str
    start_char: int
    end_char: int


def chunker(file_path:str, max_chunk_size: int, source: str) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=0, add_start_index = True)
    chunks = splitter.create_documents([source])
    res: list[Chunk] = []
    for chunk in chunks:
        start_char = chunk.metadata['start_index']
        end_char = start_char + len(chunk.page_content)
        res.append(Chunk(file_path, chunk.page_content, start_char, end_char))
    return res