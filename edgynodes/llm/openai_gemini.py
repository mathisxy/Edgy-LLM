from .base import Supports
from .openai import LLMNodeOpenAI

class LLMNodeGemini(LLMNodeOpenAI):


    supports: Supports = Supports(
        remote_image_urls=False
    )

    def __init__(self, model: str, api_key: str, base_url: str ="https://generativelanguage.googleapis.com/v1beta/openai/", enable_streaming: bool = True) -> None:
        super().__init__(model, api_key, base_url, enable_streaming)