import os

from openai import OpenAI

from .base import LanguageModel

LLAMA_ENDPOINT = os.environ.get("LLAMA_ENDPOINT", "http://127.0.0.1:8000/v1")
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY", "EMPTY")
LLAMA_TEMPERATURE = float(os.environ.get("LLAMA_TEMPERATURE", "0.3"))
LLAMA_MAX_TOKENS = int(os.environ.get("LLAMA_MAX_TOKENS", "1000"))
LLAMA_TOP_P = float(os.environ.get("LLAMA_TOP_P", "0.95"))


class LlamaServer(LanguageModel):
    def __init__(self, model: str = "Meta-Llama-3-70B-Instruct", *args, **kwargs):
        model = os.environ.get("LLAMA_MODEL", model)
        super().__init__(model, *args, **kwargs)
        self.client = OpenAI(
            base_url=LLAMA_ENDPOINT,
            api_key=LLAMA_API_KEY,
        )

    def chat(self, message: str, system_msg: str = None, json_mode: bool = False):
        if system_msg is None:
            system_msg = "You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": message},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=LLAMA_TEMPERATURE,
            max_tokens=LLAMA_MAX_TOKENS,
            top_p=LLAMA_TOP_P,
        )
        response = response.choices[0].message.content
        return response

    def complete(self, prompts: str):
        response = self.client.completions.create(
            model=self.model, prompt=prompts, echo=False, max_tokens=100
        )
        response = response.choices[0].text
        return response


if __name__ == "__main__":
    llama = LlamaServer("Meta-Llama-3-8B-Instruct")
    response = llama.complete(
        "The reason of human landing on moon is that, some one found it strange behind the moon."
    )
    print(response)
