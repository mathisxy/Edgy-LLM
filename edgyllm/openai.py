from .nodes import LLMNode
from .states import LLMGraphState

from openai import AsyncOpenAI as OpenAI

from llm_ir import AIMessage, AIRoles, AIChunkText, AIChunkImageURL
from llm_ir.adapter import to_openai, OpenAIMessage
import requests
import base64


class LLMNodeOpenAI(LLMNode):

    client: OpenAI

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        super().__init__(model)

        self.client = OpenAI(api_key=api_key, base_url=base_url)


    async def run(self, state: LLMGraphState) -> LLMGraphState:
        
        history = state.messages

        printable_history = [message for message in history if not any(isinstance(chunk, AIChunkImageURL) for chunk in message.chunks)]
        # print(printable_history)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.format_messages(history), # type: ignore
        )

        # print(response.choices[0].message.content)

        state.messages.append(
            AIMessage(
                role=AIRoles.MODEL,
                chunks=[AIChunkText(
                    text=str(response.choices[0].message.content)
                    )
                ],
            )
        )

        return state
    
    def format_messages(self, messages: list[AIMessage]) -> list[OpenAIMessage]:

        if not self.supports.remote_image_urls:

            for msg in messages:
                for chunk in msg.chunks:
                    if isinstance(chunk, AIChunkImageURL) and chunk.url.startswith("http"):
                        try:
                            response = requests.get(chunk.url)
                            # print(len(response.content))
                            response.raise_for_status()
                            image_data = response.content
                            mime_type = response.headers.get('content-type', '')
                            if not mime_type:
                                raise ValueError("Unknown MIME type")
                            # print(mime_type)
                            base64_data = base64.b64encode(image_data).decode('utf-8')
                            chunk.url = f"data:{mime_type};base64,{base64_data}"
                        except Exception as e:
                            print(f"Error downloading image from URL {chunk.url}: {e}")

        return to_openai(messages)
    