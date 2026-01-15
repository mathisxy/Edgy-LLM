from edgygraph import GraphNode
from pydantic import BaseModel
from .states import LLMGraphState


class Supports(BaseModel):

    vision: bool = True
    audio: bool = False
    streaming: bool = True
    remote_image_urls: bool = True



class LLMNode(GraphNode[LLMGraphState]):

    model: str

    supports: Supports = Supports()
    
    def __init__(self, model: str) -> None:
        self.model = model