"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
import os 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display 
from configuration import Configuration
from app.react_agent.state import OverallState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from datetime import date
import logging

# Suppress debug messages from ipywidgets
logging.getLogger('ipywidgets').setLevel(logging.WARNING)
logging.getLogger('comm').setLevel(logging.WARNING)
logging.getLogger('tornado').setLevel(logging.WARNING)
logging.getLogger('traitlets').setLevel(logging.WARNING)

#------------------------------------------------------------------------
# from agents import agent_node, retrieve_node, create_tool_node_with_fallback, 
from app.react_agent.mlb.agents import team_agent, player_agent, game_data_agent, game_info_agent
from app.react_agent.prompts import MAIN_SUPERVISOR_PROMPT
# ---------------------------------------------------------------------
# Disable all logging globally
logging.disable(logging.CRITICAL)  # Disable all logging below CRITICAL level

# Redirect all logging output to os.devnull
logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()])
# Suppress warnings as well (optional)
import warnings
warnings.filterwarnings("ignore")



llm = ChatOpenAI(model="gpt-4o")
# Create supervisor workflow
mlb_workflow = create_supervisor(
    [team_agent, player_agent, game_data_agent, game_info_agent],
    model=llm,
    prompt=MAIN_SUPERVISOR_PROMPT
)

# Compile and run
app_mlb = mlb_workflow.compile(name = "MLB_Workflow")

# display(
#     Image(
#         app.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )
