'''

1. Accept natural language queries from the user.
2. Decide which tool (or sequence of tools) to use.
3.Call the MCP server via the stdio interface.
4. Return a structured, helpful response.

> Use LangGraph to define the interaction flow as a graph of nodes.
> Each node represents a reasoning step, such as generating a response or selecting a tool.
> OpenAI’s LLM powers the agent, and LangChain’s MCP adapters allow tool integration during runtime.


'''


import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.studio import stdio_client

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated, List, TypedDict
from langchain.adapter.tools import load_mcp_tools

# The MCP server runs as a local subprocess using standard input/output (stdio) as the transport layer.
# We define the server’s command and arguments using StdioServerParameters.
server_params = StdioServerParameters(command="python", args=["mcp_server.py"])


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def create_graph(session):
    tools = await load_mcp_tools(session)
    llm = ChatOpenAI(model="gpt-4", temperature=0,
                     openai_api_key="{{OPENAI_API_KEY}}")
    llm_with_tools = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful assistant that uses tools to search the Wikipedia."), MessagesPlaceholder("messages")])

    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Build Langgraph with tool routing
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_condition_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            agent = await create_graph(session)
            print("Wikipedia agent is ready")
            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break
                try:
                    response = await agent.ainvoke(
                        {"messages": user_input},
                        config={"configurable": {"thread_id": "wiki-session"}}
                    )
                    print("AI:", response["messages"][-1].content)
                except Exception as err:
                    print("Error: ", err)


if __name__ == "__main__":
    asyncio.run(main())
