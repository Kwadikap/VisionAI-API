import os
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from .prompt import INSTRUCTION


pro_agent = LlmAgent(
    name="vision_pro",
    model="gemini-2.0-flash-exp",
    instruction=INSTRUCTION,
    description="Research-grade assistant that grounds answers with Google search and cites sources when helpful.",
    tools=[google_search],
)