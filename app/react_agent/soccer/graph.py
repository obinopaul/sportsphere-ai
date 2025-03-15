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

from configuration import Configuration
from app.react_agent.state import OverallState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

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
from app.react_agent.agents import supervisor_node, team_node, game_node, player_node, aggregator_node

# ---------------------------------------------------------------------
# Disable all logging globally
logging.disable(logging.CRITICAL)  # Disable all logging below CRITICAL level

# Redirect all logging output to os.devnull
logging.basicConfig(level=logging.CRITICAL, handlers=[logging.NullHandler()])

# Suppress warnings as well (optional)
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Any, Optional, Sequence, TypedDict, Annotated
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import StructuredTool
from nba_api.stats.endpoints import leaguestandingsv3, teamyearbyyearstats, leaguegamefinder
from nba_api.stats.static import teams
from nba_api.stats.library.parameters import SeasonTypeAllStar, SeasonYear, Season, PerModeSimple
import pandas as pd
from datetime import datetime, timedelta
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_function_calling_executor  # Correct import
from langgraph.graph import END, StateGraph
import operator
import asyncio
from app.react_agent.soccer.agents import league_supervisor, team_soccer_supervisor, player_soccer_supervisor, fixture_supervisor


# --- AgentState Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sub_queries: Optional[List[Dict[str, str]]] = None  # List of dicts: {query, supervisor}
    current_query: Optional[str] = None
    # agent_index: Not needed - we're using supervisor names directly


# --- Supervisor Prompts (from previous response - included for completeness) ---
# (Include the GAME_SUPERVISOR_PROMPT, PLAYER_SUPERVISOR_PROMPT,
#  TEAM_SUPERVISOR_PROMPT, and NBA_SUPERVISOR_PROMPT from a previous response here.)
#  For brevity, I am not putting it again.

llm = ChatOpenAI(model="gpt-4o")  # Or your preferred LLM

# --- Helper Functions ---

async def split_node(state: AgentState) -> Dict[str, List[Dict[str, str]]]:
    """Splits the user's query into sub-queries and assigns supervisors."""

    class SubQuery(BaseModel):
        query: str = Field(..., description="The sub-query text.")
        supervisor: str = Field(..., description="The name of the assigned supervisor agent ('league_supervisor', 'team_soccer_supervisor', 'player_soccer_supervisor', or 'fixture_supervisor').")

    class ParsedOutput(BaseModel):
        sub_queries: List[SubQuery] = Field(..., description="A list of sub-query dictionaries. Each dictionary MUST contain a 'query' key (the sub-query) and a 'supervisor' key (the name of the assigned supervisor). The number of sub-queries should be between 1 and 10, inclusive, based on the complexity of the original query. Simple queries should have fewer sub-queries.")


    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Or your preferred LLM
    llm_with_structure = llm.with_structured_output(ParsedOutput)

    prompt = PromptTemplate(
        template="""You are an advanced query decomposition and routing assistant for a comprehensive soccer information system. Your task is to analyze a user's query and break it down into a set of smaller, focused sub-queries, assigning each to the most appropriate supervisor agent.

        **Analyze the user's query:** `{query}`.

        **Determine the optimal number of sub-queries:** Break down the query into *1 to 10* sub-queries, depending on the complexity and scope of the original question.
            *   A simple question requiring information from only one area might only need *one* sub-query.
            *   A complex question requiring information from multiple areas, or requiring in-depth analysis, might need *several* sub-queries, *but no more than 10*.  Use your judgment and **prioritize simplicity and directness**.
            *   **Do not generate unnecessary sub-queries.** If the original query can be answered directly by one supervisor, do *not* create additional sub-queries.

        **Supervisor Agents and Their Expertise:**

        1.  **league_supervisor**: This supervisor is the expert on all things related to soccer *leagues*.  It can handle queries about:
            *   **General League Information:** League history, format, participating teams, rules, and regulations.
            *   **League Standings:**  Detailed standings for specific seasons, including various statistical breakdowns.
            *   **League Schedules:**  Retrieving schedules for entire leagues across specific date ranges.
            * **League IDs:** Can be used to retrieve league IDs

        2.  **team_soccer_supervisor**: This supervisor is the expert on all things related to soccer *teams*. It can handle queries about:
            *   **Team Fixtures:**  Past and upcoming match schedules for a specific team.
            *   **Live Match Information:** Real-time scores, statistics, and event timelines for matches *currently in progress*.
            *   **General Team Information:**  Team history, roster, stadium, and other non-match-specific details.
            *   **Team Performance Analysis:** Trends in wins/losses, goal-scoring, etc.

        3.  **player_soccer_supervisor**: This supervisor is the expert on all things related to individual soccer *players*. It can handle queries about:
            *   **Player Profiles:** Biographical information, career history, team affiliations.
            *   **Player Statistics:** Detailed statistics, filterable by season and/or league.
            *   **Player ID:** Can be used to retrieve Player IDs
            *   **Player Comparisons:**  Comparing statistics and performance of multiple players.
            *   **Player Performance Analysis:** Trends in a player's performance over time.

        4.  **fixture_supervisor**: This supervisor is the expert on *specific soccer matches (fixtures)* and *league schedules*. It can handle queries about:
            *   **Fixture Schedules:** Retrieving schedules for specific leagues and dates.
            * **Fixture Stats** Retrieving Fixture Stats
            *   **Fixture IDs:**  Finding the unique IDs for specific matches.
            *   **Detailed Fixture Statistics:** In-depth statistics for individual matches (shots, possession, passing accuracy, fouls, cards, etc.).
            *   **Fixture Analysis:** Comparing team performance within a specific match.

        **Output Format:**

        Return a JSON object with a single key, 'sub_queries'. The value is a list of dictionaries. Each dictionary MUST have the following keys:

        *   `query`: The sub-query text (string).
        *   `supervisor`: The name of the supervisor agent that should handle this sub-query (string). Must be one of: "league_supervisor", "team_soccer_supervisor", "player_soccer_supervisor", or "fixture_supervisor".

        **Key Principles:**

        *   **Directness:** Sub-queries should be as direct and to-the-point as possible.  They should be phrased as questions that the assigned supervisor can directly answer.
        *   **Specificity:** Each sub-query must be clearly answerable by *one* of the supervisors. Avoid ambiguity.
        *   **Independence:** Sub-queries should be as independent of each other as possible.
        *   **Simplicity:** Favor fewer sub-queries when possible. Avoid unnecessary decomposition.
        * **One Supervisor per Sub-query:** Each sub-query should map to only *one* supervisor.

       **EXAMPLES:**

        **Example 1 (Simple - Single Supervisor):**
        User Query: "What's the score of the Liverpool game?"
        Output: `{{"sub_queries": [{{"query": "What's the score of the Liverpool game right now?", "supervisor": "team_soccer_supervisor"}}]}}`

        **Example 2 (Advanced - Mixed Supervisors - Live and Historical Data):**
        User Query: "Is Lionel Messi playing tonight? If so, what are his career goals in Champions League?"
        Output: `{{"sub_queries": [{{"query": "Is Lionel Messi's team playing in a live match tonight?", "supervisor": "team_soccer_supervisor"}}, {{"query": "What are Lionel Messi's career goals in the Champions League?", "supervisor": "player_soccer_supervisor"}}]}}`

        **Example 3 (Advanced - Multiple Supervisors):**
        User Query: "Compare the average goals scored per game for Real Madrid and Barcelona over the last three seasons."
        Output: `{{"sub_queries": [{{"query": "What were Real Madrid's average goals scored per game for the last three seasons?", "supervisor": "team_soccer_supervisor"}}, {{"query": "What were Barcelona's average goals scored per game for the last three seasons?", "supervisor": "team_soccer_supervisor"}}]}}`

        **Example 4 (Complex - League and Player Statistics):**
        User Query: "Which team in the Premier League has the most players with over 10 goals this season, and list those players with their goal counts?"
        Output: `{{"sub_queries": [
            {{"query": "Which teams are in the Premier League?", "supervisor": "league_supervisor"}},
            {{"query": "For each team in the Premier League, which players have over 10 goals this season?", "supervisor": "player_soccer_supervisor"}},
            {{"query": "Combine the results from the previous sub-queries to determine which Premier League team has the most players with over 10 goals.", "supervisor": "league_supervisor"}}
        ]}}`

        **Example 5 (Complex - Fixture Difficulty and Team Form):**
        User Query: "Analyze Arsenal's next 5 fixtures.  Assess the difficulty of each match based on the opponent's current league standing and recent form (last 5 matches).  Also, provide Arsenal's current form (last 5 matches)."
        Output: `{{"sub_queries": [
            {{"query": "What are Arsenal's next 5 fixtures?", "supervisor": "team_soccer_supervisor"}},
            {{"query": "What is the current league standing of [opponent 1 from Arsenal's fixtures, opponent 2, etc.]?", "supervisor": "league_supervisor"}},
            {{"query": "What are the last 5 match results for [opponent 1 from Arsenal's fixtures, opponent 2, etc.]?", "supervisor": "team_soccer_supervisor"}},
            {{"query": "What are Arsenal's last 5 match results?", "supervisor": "team_soccer_supervisor"}}
        ]}}`
        
        **Example 6 (Complex - Multi-faceted Team Comparison):**
        User Query: "Analyze the upcoming match between Manchester United and Chelsea.  Compare their last 5 head-to-head results, their current league standings, their top scorers' form in the last 3 matches, and any relevant news that might impact the game."
        Output: `{{"sub_queries": [
            {{"query": "What are the last 5 head-to-head results between Manchester United and Chelsea?", "supervisor": "team_soccer_supervisor"}},
            {{"query": "What are the current league standings for Manchester United and Chelsea?", "supervisor": "league_supervisor"}},
            {{"query": "Who are the top scorers for Manchester United and Chelsea?", "supervisor": "team_soccer_supervisor"}},
            {{"query": "What are the top scorer for Manchester United's stats in their last 3 matches?", "supervisor": "player_soccer_supervisor"}},
            {{"query": "What are the top scorer for Chelsea's stats in their last 3 matches?", "supervisor": "player_soccer_supervisor"}},
            {{"query": "What is the latest news regarding the upcoming Manchester United vs. Chelsea match, including injuries, suspensions, and tactical previews?", "supervisor": "team_soccer_supervisor"}}
        ]}}`

        **Example 7 (Complex - Player Performance and Fixture Analysis):**
        User Query: "Compare Lionel Messi and Cristiano Ronaldo's performance in their last 5 matches in any competition. Include goals, assists, shots on target, and key passes. Also, analyze the difficulty of their opponents based on league standings."
        Output: `{{"sub_queries": [
            {{"query": "What are Lionel Messi's last 5 matches?", "supervisor": "player_soccer_supervisor"}},
            {{"query": "What are Cristiano Ronaldo's last 5 matches?", "supervisor": "player_soccer_supervisor"}},
            {{"query": "Get goals, assists, shots on target, and key passes for Lionel Messi in [match IDs from first sub-query, separated by commas].", "supervisor": "player_soccer_supervisor"}},
            {{"query": "Get goals, assists, shots on target, and key passes for Cristiano Ronaldo in [match IDs from second sub-query, separated by commas].", "supervisor": "player_soccer_supervisor"}},
            {{"query": "What is the current league standing of [opponent 1 from Messi's matches, opponent 2, etc.]?", "supervisor": "league_supervisor"}},
            {{"query": "What is the current league standing of [opponent 1 from Ronaldo's matches, opponent 2, etc.]?", "supervisor": "league_supervisor"}}
        ]}}`
        
        """,
        input_variables=["query"]
    )

    chain = prompt | llm_with_structure
    query = state["messages"][-1].content
    structured_output = await chain.ainvoke({"query": query})
    return {"sub_queries": [sub_query.dict() for sub_query in structured_output.sub_queries]}   # Convert to dict



async def run_supervisor(state: AgentState, supervisor_dict: Dict[str, Any]) -> AgentState:
    """Runs the appropriate supervisor based on the assigned supervisor name."""
    sub_query_info = state['current_query']
    sub_query = sub_query_info['query']
    supervisor_name = sub_query_info['supervisor']
    current_date = datetime.now().isoformat()

    # --- CRITICAL CHANGE:  Look up the supervisor by NAME ---
    supervisor = supervisor_dict[supervisor_name]

    supervisor_input = {
        "messages": state["messages"][:1] + [HumanMessage(content=f"{sub_query} Today is: {current_date}")],
    }
    response = await supervisor.ainvoke(supervisor_input)
    return {"messages": [response['messages'][-1]]}


async def parallel_runner(state: AgentState, supervisor_dict: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
    """Runs the appropriate supervisors in parallel for each sub-query."""
    tasks = []
    for sub_query_info in state["sub_queries"]:
        # Pass the entire dictionary containing query AND supervisor
        updated_state = {**state, "current_query": sub_query_info}
        task = asyncio.create_task(run_supervisor(updated_state, supervisor_dict))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    all_messages = []
    for result in results:
        all_messages.extend(result["messages"])
    return {"messages": all_messages}



async def combine_results(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """Combines results and presents to LLM for final answer."""
    final_results = [msg.content for msg in state["messages"][1:]]
    combined_results_str = "\n\n".join(final_results)

    final_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    final_prompt = PromptTemplate(
        template="""You are an expert NBA assistant.

        Original query: {original_query}

        Sub-query results: {combined_results}

        Provide a comprehensive answer.
        """,
        input_variables=["original_query", "combined_results"]
    )
    final_chain = final_prompt | final_llm
    final_answer = await final_chain.ainvoke({"original_query": state["messages"][0].content, "combined_results": combined_results_str})

    new_messages = [state["messages"][0], HumanMessage(content=final_answer.content)]
    return {"messages": new_messages}


# --- Main Entrypoint and Workflow Compilation ---
supervisor_dict = {
    "league_supervisor": league_supervisor,
    "team_soccer_supervisor": team_soccer_supervisor,
    "player_soccer_supervisor": player_soccer_supervisor,
    "fixture_supervisor": fixture_supervisor,
}
    


workflow = StateGraph(AgentState)
workflow.add_node("split_query", split_node)
workflow.add_node("parallel_supervisors", lambda state, supervisors=supervisor_dict: asyncio.run(parallel_runner(state, supervisors))) # Changed to dict
workflow.add_node("combine_results", combine_results)

workflow.add_edge(START, "split_query")
workflow.add_edge("split_query", "parallel_supervisors")
workflow.add_edge("parallel_supervisors", "combine_results")
workflow.add_edge("combine_results", END)

app_soccer = workflow.compile()


# """Main entry point."""
# query = "What's the score of the Man U game?"
# initial_state = {"messages": [HumanMessage(content=query)]}

# final_state = app_soccer.ainvoke(initial_state)



