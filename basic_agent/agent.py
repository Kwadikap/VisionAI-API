import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
# from google.adk.tools import google_search
from .prompt import INSTRUCTION


basic_agent = LlmAgent(
    name="vision_basic",
    model="gemini-2.0-flash-exp",
    instruction=INSTRUCTION,
    description="Friendly general assistant for everyday questions. Fast, concise, and tool-free.",
    tools=[],
)