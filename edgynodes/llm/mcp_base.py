from fastmcp import Client
from .base import LLMState, LLMShared
from edgygraph import Node
from llm_ir import Tool
from typing import Any
import mcp


class AddToolsFromMCPNode[T: LLMState = LLMState, S: LLMShared = LLMShared](Node[T, S]):

    client: Client[Any]

    def __init__(self, url: str) -> None:
        super().__init__()

        self.client = Client(url)

    async def run(self, state: T, shared: S) -> None:
        
        tools: list[Tool] = self.format_tools(await self.client.list_tools())

        state.tools.extend(tools)

        async with shared.lock:
            for tool in tools:
                async def function(**args: Any):
                    return await self.client.call_tool(
                        name=tool.name,
                        arguments=args
                    )
                if tool.name in shared.tool_functions:
                    raise Exception(f"Tool with name \"{tool.name}\" already exists")
                shared.tool_functions[tool.name] = function

    
    def format_tools(self, mcp_tools: list[mcp.types.Tool]) -> list[Tool]:

        tools: list[Tool] = []

        for tool in tools:
            tools.append(
                Tool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_schema,
                )
            )

        return tools
