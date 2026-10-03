from transformers import AutoModelForCausalLM, AutoTokenizer

class Small_LLM:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)

    def generate(self):
        ...