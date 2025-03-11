"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
from pydantic import BaseModel, Field
from typing import Optional
from IPython.display import Image, display
from dotenv import load_dotenv
from datetime import date
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig, RunnableLambda, Runnable
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI as LangchainChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from typing import Dict, List, Literal, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import ipywidgets as widgets
from IPython.display import display
import os 
import re
import json
from datetime import date
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, Tool
from langchain_core.tools import tool
from langchain import hub

#------------------------------------------------------------
from app.react_agent.state import OverallState
from app.react_agent.prompts import *
from app.react_agent.tools import *
from app.react_agent.configuration import *
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode


from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from app.react_agent.tools import team_tools, player_tools, game_info_tools, game_data_tools
from app.react_agent.prompts import TEAM_PROMPT, PLAYER_PROMPT, GAME_INFO_PROMPT, GAME_DATA_PROMPT, MAIN_SUPERVISOR_PROMPT, GAME_SUPERVISOR_PROMPT
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="gpt-4o")

team_agent = create_react_agent(
    model=llm,
    tools=team_tools,
    name="team_agent",
    prompt= TEAM_PROMPT
)

player_agent = create_react_agent(
    model=llm,
    tools=player_tools,
    name="player_agent",
    prompt=PLAYER_PROMPT
)

game_info_agent = create_react_agent(
    model=llm,
    tools=game_info_tools,
    name="game_info_agent",
    prompt=GAME_INFO_PROMPT
)

game_data_agent = create_react_agent(
    model=llm,
    tools=game_data_tools,
    name="game_data_agent",
    prompt=GAME_DATA_PROMPT
)


# Create supervisor workflow
mlb_workflow = create_supervisor(
    [team_agent, player_agent, game_data_agent, game_info_agent],
    model=llm,
    prompt=MAIN_SUPERVISOR_PROMPT
)

# Compile and run
app_mlb = mlb_workflow.compile(name = "MLB_Workflow")

# result = app_mlb.invoke({
#     "messages": [
#         {
#             "role": "user",
#             "content": "Tell me about the performance of Pittsburgh Pirates in the 2024 season"
#         }
#     ]
# })
