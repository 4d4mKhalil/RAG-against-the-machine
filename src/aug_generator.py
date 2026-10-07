from .llm import Small_LLM
from functools import lru_cache


@lru_cache()
def get_llm() -> Small_LLM:
    return Small_LLM(model="Qwen/Qwen3-0.6B")


def augmenter(query: str, context: list[str], token_limit: int = 200, max_chunks: int = 3) -> str:
    llm = get_llm()
    ctx = "\n\n".join(
        f"[{c.file_path}]\n{c.text}" for c in context[:max_chunks]
    )
    prompt = (
        "Answer the question using only the context. "
        "Be concise. If the context is insufficient, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )
    return str(llm.generate(prompt, max_new_tokens=token_limit))