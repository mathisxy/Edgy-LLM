from edgygraph import Node, State, Shared, Stream
from llm_ir import AIMessage
from pydantic import BaseModel, Field


class LLMState(State):
    messages: list[AIMessage] = Field(default_factory=list[AIMessage])

class LLMShared(Shared):
    llm_stream: Stream[str] | None = None
    pass


class Supports(BaseModel):

    vision: bool = True
    audio: bool = False
    streaming: bool = True
    remote_image_urls: bool = True


class LLMNode[T: LLMState, S: LLMShared](Node[T, S]):

    model: str
    
    enable_streaming: bool

    supports: Supports = Supports()
    
    def __init__(self, model: str, enable_streaming: bool = False) -> None:
        self.model = model
        self.enable_streaming = enable_streaming