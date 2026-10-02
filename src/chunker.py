import ast
from dataclasses import dataclass


@dataclass
class Chunk:
    file_path: str
    text: str
    start_char: int
    end_char: int


class PythonChunker:
    def __init__(self, max_chunk_size: int = 2000):
        self.max_chunk_size = max_chunk_size

    def _split(self, file_path: str, text: str, offset: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        i = 0
        while i < len(text):
            end = min(i + self.max_chunk_size, len(text))
            chunks.append(Chunk(
                file_path,
                text[i:end],
                offset + i,
                offset + end
                )
                )
            i = end
        return chunks

    def _tree_failure(self, file_path: str, source: str) -> list[Chunk]:
        return self._split(file_path, source, offset=0)

    def chunk(self, file_path: str, source: str) -> list[Chunk]:
        chunks: list[Chunk] = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._tree_failure(file_path, source)

        lines = source.splitlines(keepends=True)
        lines_map = [0]
        for line in lines:
            lines_map.append(lines_map[-1] + len(line))

        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef
                )
                )
        ]
        nodes.sort(key=lambda n: n.lineno)

        for node in nodes:
            if node.end_lineno is None:
                continue
            start_char = lines_map[node.lineno - 1]
            end_char = lines_map[node.end_lineno]
            text = source[start_char:end_char]
            if len(text) <= self.max_chunk_size:
                chunks.append(Chunk(file_path, text, start_char, end_char))
            else:
                chunks.extend(self._split(file_path, text, start_char))

        if not chunks:
            chunks = self._tree_failure(file_path, source)

        return chunks
