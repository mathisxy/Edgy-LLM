from llmir import Tool, AIChunkText, AIChunkToolCall, AIMessageToolResponse
from .base import LLMState, LLMShared
from edgygraph import Node
from typing import Any, Callable, Type, Tuple, cast
from pydantic import Field, create_model, BaseModel
from docstring_parser import parse
from rich import print as rprint
import json
import inspect



class AddToolsNode[T: LLMState = LLMState, S: LLMShared = LLMShared](Node[T, S]): #TODO make it make sense

    tools: dict[str, Tuple[Callable[..., Any], Tool]]

    def __init__(self, functions: list[Callable[..., Any]]) -> None:
        super().__init__()

        self.tools = self.format_functions(functions)

    async def run(self, state: T, shared: S) -> None:
        
        async with shared.lock:
            for key, value in self.tools.items():
                
                if key in shared.tool_functions:
                    raise Exception(f"Duplicate function name: {key}")
                
                function, tool = value

                shared.tool_functions[key] = function
                state.tools.append(tool)

    
    def format_functions(self, functions: list[Callable[..., Any]]) -> dict[str, Tuple[Callable[..., Any], Tool]]:

        tools: dict[str, Tuple[Callable[..., Any], Tool]] = {}

        for function in functions:

            if function.__name__ in tools:
                raise Exception(f"Duplicate function name: {function.__name__}")

            doc = parse(function.__doc__ or "")
            param_descriptions = {p.arg_name: p.description for p in doc.params}

            signature = inspect.signature(function)
            fields = {}

            for name, param in signature.parameters.items():

                # Skip *args or **kwargs
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
                default = ... if param.default is inspect.Parameter.empty else param.default

                fields[name] = (
                    annotation, 
                    Field(
                        default=default,
                        description=param_descriptions.get(name, "")
                    )
                )

            dynamic_model: Type[BaseModel] = create_model(function.__name__, **cast(dict[str, Any], fields))

            tools[function.__name__] = (
                function,
                Tool(
                    name=function.__name__,
                    description=doc.description or "",
                    input_schema=dynamic_model.model_json_schema(),
                )
            )
        
        return tools


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

                        print(f"Executed function {chunk.name}")
                        rprint(state)

                    except json.JSONDecodeError as e:
                        e.add_note(f"Unable to JSON encode result for function {chunk.name} with arguments {chunk.arguments}")
                        raise e


    async def format_result(self, chunk: AIChunkToolCall, result: Any) -> AIMessageToolResponse:

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