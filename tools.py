
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

playwright = MCPToolset(
               connection_params=StdioConnectionParams(
                   timeout_seconds=300,
                   server_params=StdioServerParameters(
                       command="npx",
                       args=[
                           "-y",
                           "@playwright/mcp@latest",
                           "--browser=chrome",
                           "--caps=pdf",
                           "--caps=vision",
                           "--headless",
                       ],
                   )
               ),
           )
