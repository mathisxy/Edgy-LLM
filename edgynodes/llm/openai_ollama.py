from .openai import LLMNodeOpenAI
from .base import Supports

class LLMNodeOllama(LLMNodeOpenAI):

    supports: Supports = Supports(
        remote_image_urls=False,
    )

    def __init__(self, model: str, api_key: str, base_url: str ="http://localhost:11434/v1", enable_streaming: bool = True) -> None:
        super().__init__(model, api_key, base_url, enable_streaming)