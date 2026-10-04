from .llm import Small_LLM
from functools import lru_cache


@lru_cache()
def get_llm() -> Small_LLM:
    return Small_LLM(model="Qwen/Qwen3-0.6B")


def augmenter(query: str, context: list[str], token_limit: int = 150) -> str:
    llm = get_llm()

    augmented_prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    return str(llm.generate(augmented_prompt, max_new_tokens=token_limit))