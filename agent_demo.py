import os
import asyncio

from dotenv import load_dotenv
from langchain_gigachat import GigaChat
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "knowledge_base": {
                "transport": "stdio",
                "command": "python",
                "args": ["local_mcp_server.py"],
            }
        }
    )

    tools = await client.get_tools()

    print("Available tools:")
    for tool in tools:
        print(tool.name)

    tools = [tool for tool in tools if tool.name == "qdrant_find"]

    print("Tools used by agent:")
    for tool in tools:
        print(tool.name)

    model = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
        temperature=0,
    )

    agent = create_react_agent(
        model=model,
        tools=tools,
    )

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
                    Найди top-3 фрагмента про universities in Korea.

                    Используй инструмент поиска.

                    Для каждого результата верни:
                    - document_id
                    - chunk_id
                    - source
                    - score
                    - text
                    """
                )
            ]
        }
    )

    print("\nAgent result:\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())