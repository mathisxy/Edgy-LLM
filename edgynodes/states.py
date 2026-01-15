from edgygraph import GraphState
from llm_ir import AIMessage
from typing import TypeVar, Generic

T = TypeVar('T', bound=object, default=object, covariant=True)

class LLMGraphState(GraphState[T], Generic[T]):
    messages: list[AIMessage] = []