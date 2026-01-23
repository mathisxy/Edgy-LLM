from .openai import LLMNodeOpenAI

class LLMNodeAzure(LLMNodeOpenAI):

    """The Base-URL should be in this format: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/"""

    def __init__(self, model: str, api_key: str, base_url: str, enable_streaming: bool = True) -> None:
        super().__init__(model=model, api_key=api_key, base_url=base_url, enable_streaming=enable_streaming)