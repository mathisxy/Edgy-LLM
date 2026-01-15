from .openai import LLMNodeOpenAI

class LLMNodeMistral(LLMNodeOpenAI):

    def __init__(self, model: str, api_key: str, base_url: str ="https://api.mistral.ai/v1", enable_streaming: bool = True) -> None:
        super().__init__(model, api_key, base_url, enable_streaming)