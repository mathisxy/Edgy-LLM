from edgygraph import Node, State, Shared, Stream
from llmir import AIMessage, Tool
from pydantic import BaseModel, Field
from typing import Callable, Any
import asyncio

class LLMStream(Stream[str]):
    abort: asyncio.Event

    def __init__(self) -> None:
        self.abort = asyncio.Event()

class LLMState(State):
    messages: list[AIMessage] = Field(default_factory=list[AIMessage])
    new_messages: list[AIMessage] = Field(default_factory=list[AIMessage])

    tools: list[Tool] = Field(default_factory=list[Tool])


class LLMShared(Shared):
    llm_stream: LLMStream | None = None

    tool_functions: dict[str, Callable[..., Any]] = Field(default_factory=dict) # name -> function
    


class Supports(BaseModel):

    vision: bool = True
    audio: bool = False
    streaming: bool = True
    remote_image_urls: bool = True


class LLMNode[T: LLMState = LLMState, S: LLMShared = LLMShared](Node[T, S]):

    model: str
    enable_streaming: bool = False

    supports: Supports = Supports()

    def __init__(self, model: str, enable_streaming: bool = False) -> None:
        super().__init__()

        self.model = model
        self.enable_streaming=enable_streaming



class AddMessageNode[T: LLMState = LLMState, S: LLMShared = LLMShared](Node[T, S]):

    message: AIMessage

    def __init__(self, message: AIMessage) -> None:
        super().__init__()

        self.message = message

    async def run(self, state: T, shared: S) -> None:
    
        state.messages.append(
            self.message
        )


class SaveNewMessagesNode[T: LLMState = LLMState, S: LLMShared = LLMShared](Node[T, S]):

    async def run(self, state: T, shared: S) -> None:
        
        state.messages.extend(state.new_messages)

        state.new_messages = []