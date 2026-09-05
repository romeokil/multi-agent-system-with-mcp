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

async def main():
    tools = await client.get_tools()

    print("\nAvailable MCP Tools:\n")

    for tool in tools:
        print(tool.name)


if __name__ == "__main__":
    asyncio.run(main())
