from edgygraph import GraphNode
from pydantic import BaseModel
from typing import TypeVar, Generic
from .states import LLMGraphState


class Supports(BaseModel):

    vision: bool = True
    audio: bool = False
    streaming: bool = True
    remote_image_urls: bool = True

T = TypeVar('T', bound=LLMGraphState)

class LLMNode(GraphNode[T], Generic[T]):

    model: str
    
    enable_streaming: bool

    supports: Supports = Supports()
    
    def __init__(self, model: str, enable_streaming: bool = False) -> None:
        self.model = model
        self.enable_streaming = enable_streaming