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
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, Tool
from IPython.display import Image, display
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


llm = ChatOpenAI(model="gpt-4o")
# -------------------------------- League --------------------------------

league_info_agent = create_react_agent(
    model=llm,
    tools=[get_league_info, get_all_leagues_id, get_league_id_by_name],
    name="league_info_agent",
    prompt= LEAGUE_INFO_PROMPT
)

league_schedule_standings_agent = create_react_agent(
    model=llm,
    tools=[get_league_id_by_name, get_all_leagues_id, get_standings, get_league_schedule_by_date],
    name="league_schedule_standings_agent",
    prompt=LEAGUE_SCHEDULE_STANDINGS_PROMPT
)

tavily_search_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_tool],
    name="tavily_search_agent",
    prompt=TAVILY_SEARCH_PROMPT
)


# Create supervisor for games
league_supervisor = create_supervisor(
    [league_info_agent, league_schedule_standings_agent, tavily_search_agent],
    supervisor_name = "league_supervisor",
    model=llm,
    prompt=LEAGUE_SUPERVISOR_PROMPT
).compile(name = "league_supervisor")



# -------------------------------- TEAM --------------------------------

tavily_search_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_tool],
    name="tavily_search_agent",
    prompt=TAVILY_SEARCH_PROMPT
)

live_match_agent = create_react_agent(
    model=llm,
    tools=[get_live_match_for_team, get_live_stats_for_team, get_live_match_timeline],
    name="live_match_agent",
    prompt= LIVE_MATCH_PROMPT
)

team_fixtures_agent = create_react_agent(
    model=llm,
    tools=[get_team_fixtures, get_team_fixtures_by_date_range, get_team_info],
    name="team_fixtures_agent",
    prompt=TEAM_FIXTURES_PROMPT
)

# Create supervisor for games
team_soccer_supervisor = create_supervisor(
    [tavily_search_agent, team_fixtures_agent, live_match_agent],
    supervisor_name = "team_soccer_supervisor",
    model=llm,
    prompt=TEAM_SOCCER_SUPERVISOR_PROMPT
).compile(name = "team_soccer_supervisor")


# -------------------------------- PLAYERS--------------------------------


tavily_search_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_tool],
    name="tavily_search_agent",
    prompt=TAVILY_SEARCH_PROMPT
)

player_id_stats_agent = create_react_agent(
    model=llm,
    tools=[get_player_id, get_player_statistics, get_player_profile],  # add get_league_id_by_name for get_player_statistics_2
    name="player_id_stats_agent",
    prompt=PLAYER_ID_STATS_PROMPT
)


player_soccer_stats_agent_2 = create_react_agent(
    model=llm,
    tools=[get_player_id, get_league_id_by_name, get_player_statistics_2],  # add get_league_id_by_name for get_player_statistics_2
    name="player_soccer_stats_agent_2",
    prompt=PLAYER_SOCCER_STATS_PROMPT_2
)

# Create supervisor for games
player_soccer_supervisor = create_supervisor(
    [tavily_search_agent, player_id_stats_agent, player_soccer_stats_agent_2],
    supervisor_name = "player_soccer_supervisor",
    model=llm,
    prompt=PLAYER_SOCCER_SUPERVISOR_PROMPT
).compile(name = "player_soccer_supervisor")




# -------------------------------- FIXTURES --------------------------------

live_match_agent = create_react_agent(
    model=llm,
    tools=[get_live_match_for_team, get_live_stats_for_team, get_live_match_timeline],
    name="live_match_agent",
    prompt= LIVE_MATCH_PROMPT
)

fixture_schedule_agent = create_react_agent(
    model=llm,
    tools=[get_league_schedule_by_date, get_multiple_fixtures_stats, tavily_search_tool],
    name="fixture_schedule_agent",
    prompt=FIXTURE_SCHEDULE_PROMPT
)

team_fixtures_agent = create_react_agent(
    model=llm,
    tools=[get_team_fixtures, get_team_fixtures_by_date_range, get_team_info],
    name="team_fixtures_agent",
    prompt=TEAM_FIXTURES_PROMPT
)

tavily_search_agent = create_react_agent(
    model=llm,
    tools=[tavily_search_tool],
    name="tavily_search_agent",
    prompt=TAVILY_SEARCH_PROMPT
)

# Create supervisor for games
fixture_supervisor = create_supervisor(
    [live_match_agent, fixture_schedule_agent, team_fixtures_agent, tavily_search_agent],
    supervisor_name = "fixture_supervisor",
    model=llm,
    prompt=FIXTURE_SUPERVISOR_PROMPT
).compile(name = "fixture_supervisor")


# -------------------------------- MAIN SUPERVISOR --------------------------------
main_soccer_supervisor = create_supervisor(
    [league_supervisor, team_soccer_supervisor, player_soccer_supervisor, fixture_supervisor],
    supervisor_name = "main_soccer_supervisor",
    model=llm,
    prompt=SOCCER_SUPERVISOR_PROMPT
).compile(name = "main_soccer_supervisor")

# display(
#     Image(
#         main_soccer_supervisor.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#         )
#     )
# )



