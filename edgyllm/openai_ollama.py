from llm_ir import AIMessage, AIChunkImageURL
from llm_ir.adapter import OpenAIMessage
import requests
import base64
from .openai import LLMNodeOpenai

class LLMNodeOllama(LLMNodeOpenai):

    def __init__(self, model: str, api_key: str, base_url: str ="http://localhost:11434/v1") -> None:
        super().__init__(model, api_key, base_url)

    def format_messages(self, messages: list[AIMessage]) -> list[OpenAIMessage]:

        for msg in messages:
            for chunk in msg.chunks:
                if isinstance(chunk, AIChunkImageURL):
                    try:
                        response = requests.get(chunk.url)
                        print(len(response.content))
                        response.raise_for_status()  # Wirft einen Fehler, falls die Anfrage fehlschlägt
                        image_data = response.content
                        mime_type = response.headers.get('content-type', '')
                        if not mime_type:
                            raise ValueError("Unknown MIME type")
                        print(mime_type)
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                        chunk.url = f"data:{mime_type};base64,{base64_data}"
                    except Exception as e:
                        print(f"Error downloading image from URL {chunk.url}: {e}")                    

        return super().format_messages(messages)