from types import TracebackType
from .base import LLMNode, LLMState, LLMShared, LLMStream

from openai import AsyncOpenAI as OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionFunctionToolParam, ChatCompletion

from typing import AsyncIterator, AsyncGenerator

from llmir import AIMessages, AIRoles, AIChunkText, AIChunkImageURL, Tool, AIChunks, AIChunkToolCall, AIMessage
from llmir.adapter import to_openai, OpenAIMessages
import requests
import base64
import json

class OpenAIStream(LLMStream):

    iterator: AsyncGenerator[ChatCompletionChunk]

    def __init__(self, iterator: AsyncGenerator[ChatCompletionChunk]) -> None:
        super().__init__() 
        
        self.iterator = iterator

    async def __anext__(self) -> str:

        if self.abort.is_set():
            print("Aborting Stream")
            raise StopAsyncIteration

        chunk = await self.iterator.__anext__()

        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            return text
                
        return ""
    
    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        return await super().__aexit__(exc_type, exc, tb)
    
    async def aclose(self) -> None:
        await super().aclose()
        await self.iterator.close() # type:ignore 
        print("aclose called")
        pass
        

class LLMNodeOpenAI[T: LLMState = LLMState, S: LLMShared = LLMShared](LLMNode[T, S]):

    client: OpenAI  

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", enable_streaming: bool = False) -> None:
        super().__init__(model, enable_streaming)

        self.client = OpenAI(api_key=api_key, base_url=base_url)


    async def run(self, state: T, shared: S) -> None:
        
        chat = state.messages

        # printable_history = [message for message in history if not any(isinstance(chunk, AIChunkImageURL) for chunk in message.chunks)]
        # print(printable_history)

        if not self.enable_streaming:

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.format_messages(chat), # type: ignore
                tools=self.format_tools(state.tools),
            )

            # print(response.choices[0].message.content)

            state.new_messages.append(
                self.format_response(state, response)
            )
            

        else:

            assert self.supports.streaming == True

            stream: AsyncIterator[ChatCompletionChunk] = await self.client.chat.completions.create( # type: ignore
                model=self.model,
                messages=self.format_messages(chat), # type: ignore
                tools=self.format_tools(state.tools),
                stream=True
            )

            async with shared.lock:
                if shared.llm_stream is not None:
                    raise Exception("Stream variable in Shared is occupied")
            
                shared.llm_stream = OpenAIStream(iterator=stream) # type: ignore


    def format_response(self, state: T, response: ChatCompletion):

        chunks: list[AIChunks] = []

        if tool_calls := response.choices[0].message.tool_calls:
            for tool_call in tool_calls:
                chunks.append(
                    AIChunkToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name, #type:ignore
                        arguments=json.loads(tool_call.function.arguments)#type:ignore
                    )
                )
        
        if text := response.choices[0].message.content:
            chunks.append(
                AIChunkText(
                    text=text
                )
            )
        
        return AIMessage(
            role=AIRoles.MODEL,
            chunks=chunks
        )
    
        

    def format_tools(self, tools: list[Tool]) -> list[ChatCompletionFunctionToolParam]:

        formatted_tools: list[ChatCompletionFunctionToolParam] = []

        for tool in tools:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })

        return formatted_tools

    
    def format_messages(self, messages: list[AIMessages]) -> list[OpenAIMessages]:

        if not self.supports.remote_image_urls:

            for msg in messages:
                if isinstance(msg, AIMessage):
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

        formatted = to_openai(messages)
        print(formatted)
        return formatted
    