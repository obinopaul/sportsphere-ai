"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from datetime import date
from langchain_core.messages import AIMessage
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig, RunnableLambda, Runnable
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
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
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles


from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from app.react_agent.tools import team_tools, player_tools, game_info_tools, game_data_tools
from app.react_agent.prompts import (LIVE_GAME_PROMPT, GAME_SCHEDULING_PROMPT, TEAM_GAME_LOGS_PROMPT, GAME_ONLINE_PROMPT, 
                                     GAME_SUPERVISOR_PROMPT, PLAYER_INFO_PROMPT, PLAYER_STATS_PROMPT, PLAYER_ONLINE_PROMPT,
                                     PLAYER_SUPERVISOR_PROMPT, TEAM_GAME_LOGS_PROMPT, TEAM_STATS_PROMPT, TEAM_ONLINE_PROMPT, 
                                     TEAM_SUPERVISOR_PROMPT, NBA_SUPERVISOR_PROMPT,
                                     )
from app.react_agent.prompts import *
from app.react_agent.tools import *
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="gpt-4o")


# -------------------------------- GAMES--------------------------------
live_game_agent = create_react_agent(
    model=llm,
    tools=[nba_live_scoreboard, nba_live_boxscore , nba_live_play_by_play],
    name="live_game_agent",
    prompt= LIVE_GAME_PROMPT
)

game_scheduling_agent = create_react_agent(
    model=llm,
    tools=[nba_list_todays_games, nba_live_scoreboard, tavily_search_tool],
    name="game_scheduling_agent",
    prompt=GAME_SCHEDULING_PROMPT
)

team_game_logs_agent = create_react_agent(
    model=llm,
    tools=[nba_team_game_logs, nba_team_game_logs_by_name, nba_fetch_game_results], # add nba_fetch_game_results
    name="team_game_logs_agent",
    prompt=TEAM_GAME_LOGS_PROMPT
)

game_online_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_tool],
    name="game_online_agent",
    prompt=GAME_ONLINE_PROMPT
)

# Create supervisor for games
game_supervisor = create_supervisor(
    [live_game_agent, game_scheduling_agent, team_game_logs_agent, game_online_agent],
    supervisor_name = "game_supervisor",
    model=llm,
    prompt=GAME_SUPERVISOR_PROMPT
).compile(name = "game_supervisor")


# -------------------------------- PLAYERS--------------------------------

player_info_agent = create_react_agent(
    model=llm,
    tools=[nba_search_players, nba_common_player_info, nba_list_active_players],
    name="player_info_agent",
    prompt= PLAYER_INFO_PROMPT
)

player_stats_agent = create_react_agent(
    model=llm,
    tools=[nba_search_players, nba_player_career_stats, nba_player_game_logs],
    name="player_stats_agent",
    prompt=PLAYER_STATS_PROMPT
)

player_online_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_tool],
    name="player_online_agent",
    prompt=PLAYER_ONLINE_PROMPT
)

# Create supervisor for games
player_supervisor = create_supervisor(
    [player_info_agent, player_stats_agent, player_online_agent],
    supervisor_name = "player_supervisor",
    model=llm,
    prompt=PLAYER_SUPERVISOR_PROMPT
).compile(name = "player_supervisor")


# -------------------------------- TEAMS --------------------------------


team_game_logs_agent = create_react_agent(
    model=llm,
    tools=[nba_team_game_logs, nba_team_game_logs_by_name, nba_fetch_game_results], # add nba_fetch_game_results
    name="team_game_logs_agent",
    prompt=TEAM_GAME_LOGS_PROMPT
)

team_stats_agent = create_react_agent(
    model=llm,
    tools=[nba_team_standings, nba_team_stats_by_name, nba_all_teams_stats],
    name="team_stats_agent",
    prompt=TEAM_STATS_PROMPT
)

team_online_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_tool],
    name="team_online_agent",
    prompt=TEAM_ONLINE_PROMPT
)

# Create supervisor for games
teams_supervisor = create_supervisor(
    [team_game_logs_agent, team_online_agent, team_stats_agent],
    supervisor_name = "teams_supervisor",
    model=llm,
    prompt=TEAM_SUPERVISOR_PROMPT
).compile(name = "teams_supervisor")



# -------------------------------- MAIN SUPERVISOR --------------------------------
main_supervisor = create_supervisor(
    [game_supervisor, player_supervisor, teams_supervisor],
    supervisor_name = "main_supervisor",
    # output_mode = "last_message",
    model=llm,
    prompt=NBA_SUPERVISOR_PROMPT
).compile(name = "main_supervisor")

# display(
#     Image(
#         main_supervisor.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )
