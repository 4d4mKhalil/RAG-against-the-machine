import pathlib

from .settings import Settings
from .models import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

def _chunker(file_path:str, max_chunk_size: int) -> list[Chunk]:
    extension = pathlib.Path(file_path).suffix
    language_map = {
        ".py": Language.PYTHON,
        ".md": Language.MARKDOWN,
        ".rst": Language.RST
    }
    if extension in language_map:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_map[extension],
            chunk_size=max_chunk_size,
            chunk_overlap=0,
            add_start_index = True
            )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=0, add_start_index = True)

    try:
        with open(file_path) as file:
            source = file.read()
    except IOError:
        return []
    
    chunks = splitter.create_documents([source])
    res: list[Chunk] = []
    for chunk in chunks:
        start_char = chunk.metadata['start_index']
        end_char = start_char + len(chunk.page_content)
        res.append(Chunk(
            file_path=file_path,
            first_character_index=start_char,
            last_character_index=end_char,
            text=chunk.page_content
            ))
    return res

def loader(repo: str, max_chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    files = list(pathlib.Path(repo).rglob('*'))

    for file in files:
        if not file.is_file() or Settings.is_ignored(file):
            continue
        if not Settings.is_indexable(file):
            continue
        chunks.extend(_chunker(str(file), max_chunk_size))
    return chunks