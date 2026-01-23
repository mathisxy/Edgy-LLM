from types import TracebackType
from .base import LLMNode, LLMState, LLMShared

from openai import AsyncOpenAI as OpenAI
from openai.types.chat import ChatCompletionChunk

from typing import AsyncIterator, Callable

from llm_ir import AIMessage, AIRoles, AIChunkText, AIChunkImageURL
from llm_ir.adapter import to_openai, OpenAIMessage
from edgygraph import Stream
import requests
import base64

class OpenAIStream(Stream[str]):

    iterator: AsyncIterator[ChatCompletionChunk]

    def __init__(self, iterator: AsyncIterator[ChatCompletionChunk], on_complete: Callable[[str], None] | None = None) -> None:
        
        self.iterator = iterator
        self.on_complete = on_complete


    async def __anext__(self) -> str:

        chunk = await self.iterator.__anext__()

        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            return text
                
        return ""
    
    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        return await super().__aexit__(exc_type, exc, tb)
    
    async def aclose(self) -> None:
        pass
        

class LLMNodeOpenAI[T: LLMState = LLMState, S: LLMShared = LLMShared](LLMNode[T, S]):

    client: OpenAI  

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", enable_streaming: bool = False) -> None:
        super().__init__(model, enable_streaming)

        self.client = OpenAI(api_key=api_key, base_url=base_url)


    async def run(self, state: T, shared: S) -> None:
        
        history = state.messages

        # printable_history = [message for message in history if not any(isinstance(chunk, AIChunkImageURL) for chunk in message.chunks)]
        # print(printable_history)

        if not self.enable_streaming:

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.format_messages(history), # type: ignore
            )

            # print(response.choices[0].message.content)

            state.messages.append(
                AIMessage(
                    role=AIRoles.MODEL,
                    chunks=[AIChunkText(
                        text=str(response.choices[0].message.content) # type: ignore
                        )
                    ],
                )
            )
            

        else:

            assert self.supports.streaming == True

            stream: AsyncIterator[ChatCompletionChunk] = await self.client.chat.completions.create( # type: ignore
                model=self.model,
                messages=self.format_messages(history), # type: ignore
                stream=True
            )

            if shared.llm_stream is not None:
                raise Exception("Stream variable in Shared is occupied")
            
            shared.llm_stream = OpenAIStream(iterator=stream) # type: ignore

        

    
    def format_messages(self, messages: list[AIMessage]) -> list[OpenAIMessage]:

        if not self.supports.remote_image_urls:

            for msg in messages:
                for chunk in msg.chunks:
                    if isinstance(chunk, AIChunkImageURL) and chunk.url.strip().startswith("http"):
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
    