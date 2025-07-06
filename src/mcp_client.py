import asyncio
import os
import shlex
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.tools import Tool

# Custom implementation for MCP tools loading
async def load_mcp_tools(session):
    """Load MCP tools from the session and convert them to LangChain tools"""
    # Get available tools from MCP session
    tools_response = await session.list_tools()

    langchain_tools = []
    for tool in tools_response.tools:
        def create_tool_func(tool_name):
            async def tool_func(input_data: str) -> str:
                try:
                    result = await session.call_tool(tool_name, {"query": input_data})
                    return str(result.content[0].text if result.content else "No result")
                except Exception as e:
                    return f"Error calling tool {tool_name}: {str(e)}"
            return tool_func

        lc_tool = Tool(
            name=tool.name,
            description=tool.description or f"Tool: {tool.name}",
            func=create_tool_func(tool.name)
        )
        langchain_tools.append(lc_tool)

    return langchain_tools

# MCP server launch config
server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

# LangGraph state definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


async def create_graph(session):
    tools = await load_mcp_tools(session)
    llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm_with_tools = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses tools to explore Wikipedia."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

async def list_prompts(session):
    prompt_response = await session.list_prompts()

    if not prompt_response or not prompt_response.prompts:
        print("No prompts found on the server.")
        return

    print("\nAvailable Prompts and Argument Structure:")
    for p in prompt_response.prompts:
        print(f"\nPrompt: {p.name}")
        if p.arguments:
            for arg in p.arguments:
                print(f"  - {arg.name}")
        else:
            print("  - No arguments required.")
    print("\nUse: /prompt <prompt_name> \"arg1\" \"arg2\" ...")


async def handle_prompt(session, tools, command, agent):
    parts = shlex.split(command.strip())
    if len(parts) < 2:
        print("Usage: /prompt <name> \"args>\"")
        return

    prompt_name = parts[1]
    args = parts[2:]

    try:
        # Get available prompts
        prompt_def = await session.list_prompts()
        match = next((p for p in prompt_def.prompts if p.name == prompt_name), None)
        if not match:
            print(f"Prompt '{prompt_name}' not found.")
            return

        # Check arg count
        if len(args) != len(match.arguments):
            expected = ", ".join([a.name for a in match.arguments])
            print(f"Expected {len(match.arguments)} arguments: {expected}")
            return

        # Build argument dict
        arg_values = {arg.name: val for arg, val in zip(match.arguments, args)}
        response = await session.get_prompt(prompt_name, arg_values)
        prompt_text = response.messages[0].content.text
        
        # Execute the prompt via the agent
        agent_response = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt_text)]},
            config={"configurable": {"thread_id": "wiki-session"}}
        )
        print("\n=== Prompt Result ===")
        print(agent_response["messages"][-1].content)

    except Exception as e:
        print("Prompt invocation failed:", e)


# Entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = await create_graph(session)

            print("Wikipedia MCP agent is ready.")
            print("Type a question or use the following templates:")
            print("  /prompts                - to list available prompts")
            print("  /prompt <name> \"args\"   - to run a specific prompt")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break
                elif user_input.startswith("/prompts"):
                    await list_prompts(session)
                    continue
                elif user_input.startswith("/prompt"):
                    await handle_prompt(session, tools, user_input, agent)
                    continue

                try:
                    response = await agent.ainvoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config={"configurable": {"thread_id": "wiki-session"}}
                    )
                    print("AI:", response["messages"][-1].content)
                except Exception as e:
                    print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())