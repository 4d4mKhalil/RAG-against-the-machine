from transformers import AutoModelForCausalLM, AutoTokenizer


class Small_LLM:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype="auto",
            device_map="auto"
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

        response = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True
        )

        return response

