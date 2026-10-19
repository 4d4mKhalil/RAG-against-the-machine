from .settings import Settings
from .models import Chunk
from .llm import Small_LLM


def get_llm() -> Small_LLM:
    return Small_LLM(model=Settings.LLM_MODEL)


def augmenter(
    query: str,
    context: list[Chunk],
    token_limit: int = Settings.MAX_NEW_TOKENS,
    max_chunks: int = 3
) -> str:
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
