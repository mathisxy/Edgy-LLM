from llm_ir import Tool, AIChunkText, AIChunkToolCall, AIMessageToolResponse, AIMessage
from .base import LLMState, LLMShared
from edgygraph import Node
from typing import Any
import json
import inspect



class AddToolsNode[T: LLMState = LLMState, S: LLMShared = LLMShared](Node[T, S]): #TODO make it make sense

    tools: list[Tool]

    def __init__(self, tools: list[Tool]) -> None:
        super().__init__()

        self.tools = tools

    async def run(self, state: T, shared: S) -> None:
        
        state.tools.extend(self.tools)


class HandleToolCallsNode[T: LLMState = LLMState, S: LLMShared = LLMShared](Node[T, S]):

    async def run(self, state: T, shared: S) -> None:
        
        for message in state.new_messages:

            for chunk in message.chunks:

                if isinstance(chunk, AIChunkToolCall):
                    
                    try:
                        result = await self.run_function(shared, chunk)

                        state.new_messages.append(
                            await self.format_result(chunk, result)
                        )

                    except json.JSONDecodeError as e:
                        e.add_note(f"Unable to JSON encode result for function {chunk.name} with arguments {chunk.arguments}")
                        raise e


    async def format_result(self, chunk: AIChunkToolCall, result: Any) -> AIMessage:

        if not isinstance(result, str):
            result = json.dumps(result)

        return AIMessageToolResponse(
            id=chunk.id,
            name=chunk.name,
            chunks=[
                AIChunkText(
                    text=result,
                )
            ]
        )
    
    async def run_function(self, shared: S, chunk: AIChunkToolCall) -> Any:
        
        async with shared.lock:

            func = shared.tool_functions[chunk.name]

            if inspect.iscoroutinefunction(func):
                return await func(**chunk.arguments)
            else:
                return func(**chunk.arguments)