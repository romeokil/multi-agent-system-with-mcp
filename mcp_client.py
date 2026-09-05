import os
import asyncio

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# initializing the mcp client

client = MultiServerMCPClient(
    {
        "tavily_web_search":{
            "transport": "http",
            "url":f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        }
    }
)

# ab mcp-server , mcp-client se puchega ki kon kon sa tools hai apke pass
# tools discovery

# async def main():
#     tools = await client.get_tools()

#     print("\nAvailable MCP Tools:\n")

#     for tool in tools:
#         print(tool.name)


# modifying the main function

async def main():

    tools = await client.get_tools()
    print("client_get_tools", tools)
    search_tool = next(
        tool
        for tool in tools
        if tool.name == "tavily_search"
    )

    result = await search_tool.ainvoke(
        {
            "query":"Best hotels in Delhi"
        }
    )

    print(result)

if __name__ == "__main__":
    asyncio.run(main())
