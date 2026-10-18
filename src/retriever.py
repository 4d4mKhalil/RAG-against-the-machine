import bm25s, faiss, json, re
from .settings import Settings
from .models import Chunk, MinimalSource
from functools import lru_cache
from sentence_transformers import SentenceTransformer

@lru_cache
def bm25_index_loader(index_path: str):
    return bm25s.BM25.load(index_path)

@lru_cache
def faiss_index_loader(index_path: str):
    return faiss.read_index(index_path)

@lru_cache
def chunks_loader(chunks_dir: str):
    with open(f'{chunks_dir}/{Settings.CHUNKS_NAME}') as file:
        return json.load(file)

@lru_cache
def get_embedder():
    return SentenceTransformer(Settings.EMBEDDING_MODEL)

def _txt_processor(text):
    text = text.replace('_', ' ')
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return text.lower()

def bm25_retriever(query: str, k: int, path) -> list[int]:
    bm25_index = bm25_index_loader(f'{path}/{Settings.BM25_NAME}')

    processed_query = _txt_processor(query)
    tokenized_query = bm25s.tokenize([processed_query])
    docs_i, _scores = bm25_index.retrieve(tokenized_query, k=k)

    return [int(x) for x in docs_i[0]]

def faiss_retriever(query, k, path) -> list[int]:
    embedder= get_embedder()
    embedded_query = embedder.encode([query])

    index = faiss_index_loader(f'{path}/{Settings.FAISS_NAME}')
    _scores, docs_i = index.search(embedded_query, k)

    return [int(x) for x in docs_i[0] if x >= 0]

def hybrid_search_retriever(
    query: str,
    k: int,
    path: str,
    rrf_k: int = Settings.RRF_K
) -> list[int]:

    top_k_bm25 = bm25_retriever(query, k, path)
    top_k_faiss = faiss_retriever(query, k, path)


    rrf_scores: dict[int, float] = {}
    for rank, chunk_index in enumerate(top_k_bm25, start=1):
        rrf_scores[chunk_index] = rrf_scores.get(chunk_index, 0) + 1 / (rrf_k + rank)
    for rank, chunk_index in enumerate(top_k_faiss, start=1):
        rrf_scores[chunk_index] = rrf_scores.get(chunk_index, 0) + 1 / (rrf_k + rank)

    ranked_keys = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    return ranked_keys[:k]


def searcher(query, k, path):
    indexes = hybrid_search_retriever(query, k, path)
    chunks = chunks_loader(path)
    res: list[Chunk] = []

    for index in indexes:
        chunk = chunks[index]
        res.append(
            Chunk(
                file_path=chunk['file_path'],
                first_character_index=chunk['first_character_index'],
                last_character_index=chunk['last_character_index'],
                text=chunk['text']
                )
            )
    return res