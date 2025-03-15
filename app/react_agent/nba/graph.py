"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast
import os 
from pydantic import BaseModel, Field
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

from app.react_agent.nba.agents import game_supervisor, player_supervisor, teams_supervisor

from typing import List, Dict, Any, Optional, Sequence, TypedDict, Annotated
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

# --- Helper Functions ---

async def split_node(state: AgentState) -> Dict[str, List[Dict[str, str]]]:
    """Splits the user's query into sub-queries and assigns supervisors."""

    class SubQuery(BaseModel):
        query: str = Field(..., description="The sub-query text.")
        supervisor: str = Field(..., description="The name of the assigned supervisor agent ('game_supervisor', 'player_supervisor', or 'teams_supervisor').")

    class ParsedOutput(BaseModel):
        sub_queries: List[SubQuery] = Field(..., description="A list of sub-query dictionaries.  Each dictionary MUST contain a 'query' key (the sub-query) and a 'supervisor' key (the name of the assigned supervisor). The number of sub-queries should be between 1 and 7, inclusive, based on the complexity of the original query.  Simple queries should have fewer sub-queries.")


    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_structure = llm.with_structured_output(ParsedOutput)

    prompt = PromptTemplate(
        template="""You are an advanced query decomposition and routing assistant for an NBA information system. Your task is to analyze a user's query and break it down into a set of smaller, focused sub-queries, assigning each to the most appropriate supervisor agent.

        **Analyze the user's query:** `{query}`.

        **Determine the optimal number of sub-queries:** Break down the query into *1 to 7* sub-queries, depending on the complexity and scope of the original question.
            *   A simple question requiring information from only one area might only need *one* sub-query.
            *   A complex question requiring information from multiple areas, or requiring in-depth analysis, might need *several* sub-queries, *but no more than 7*.  Use your judgment and **prioritize simplicity and directness**.
            *   **Do not generate unnecessary sub-queries.** If the original query can be answered directly by one supervisor, do *not* create additional sub-queries.

        **Supervisor Agents and Their Expertise:**

        1.  **game_supervisor**: This supervisor is the expert on all things related to NBA *games*.  It can handle queries about:
            *   **Live Games:**  Current scores, real-time statistics, play-by-play information.
            *   **Game Schedules:**  Past, present, and future game schedules, including dates, times, and opponents.
            *   **Team Game Logs:**  Detailed historical records of a specific team's games for a given season, including results (wins/losses), opponents, and dates. It does *not* handle individual player stats within those games (that's for the player_supervisor).
            *   **General Game Information:** Overall league standings, playoff brackets.  It is *not* the best choice for in-depth analysis of team *strategy* or *ownership* (that would be teams_supervisor).

        2.  **player_supervisor**: This supervisor is the expert on all things related to individual NBA *players*. It can handle queries about:
            *   **Player Biographical Information:** Height, weight, date of birth, current team, position, jersey number, college, etc.
            *   **Player Career Statistics:**  Points, rebounds, assists, steals, blocks, field goal percentage, three-point percentage, free throw percentage, etc. (per game, totals, career averages).  This includes regular season, playoffs, and All-Star games.
            *   **General Player Information**: This can include using web search to retrieve player news.

        3.  **teams_supervisor**: This supervisor is the expert on all things related to NBA *teams* (but *not* individual games, which are handled by the game_supervisor). It can handle queries about:
            *   **Team Game Logs:** A specific team's game history for a given season.  This includes dates, opponents, and results (wins/losses).
            * **General Team Information (via web search):**  Team news, ownership, coaching staff, arena information, and other details not directly related to game logs or individual player stats.
            *   **Team Statistics**: Overall league standings, specific team stats for a season (or multiple seasons), comparisons between teams, team records, and conference/division standings.


        **Output Format:**

        Return a JSON object with a single key, 'sub_queries'. The value is a list of dictionaries. Each dictionary MUST have the following keys:

        *   `query`: The sub-query text (string).
        *   `supervisor`: The name of the supervisor agent that should handle this sub-query (string). Must be one of: "game_supervisor", "player_supervisor", or "teams_supervisor".

        **Key Principles:**

        *   **Directness:**  Sub-queries should be as direct and to-the-point as possible.
        *   **Specificity:** Each sub-query must be clearly answerable by *one* of the supervisors.
        *   **Independence:** Sub-queries should be as independent of each other as possible.
        *   **Simplicity:**  Favor fewer sub-queries when possible.  Avoid unnecessary decomposition.

        **EXAMPLES:**

        **Example 1 (Simple - Single Supervisor):**
        User Query: "What's the score of the Lakers game?"
        Output: `{{"sub_queries": [{{"query": "What's the score of the Lakers game right now?", "supervisor": "game_supervisor"}}]}}`

        **Example 2 (Intermediate - Two Supervisors):**
        User Query: "What's LeBron James' height and current team?"
        Output: `{{"sub_queries": [{{"query": "What is LeBron James' height?", "supervisor": "player_supervisor"}}, {{"query": "What is LeBron James' current team?", "supervisor": "player_supervisor"}}]}}`

        **Example 3 (Advanced - Mixed Supervisors - Live and Historical Data):**
        User Query: "Is LeBron James playing tonight? If so, what are his career playoff averages?"
        Output: `{{"sub_queries": [{{"query": "Is LeBron James playing in tonight's game?", "supervisor": "game_supervisor"}}, {{"query": "What are LeBron James' career playoff averages?", "supervisor": "player_supervisor"}}]}}`
        
        **Example 4 (Advanced - Multiple Supervisors, but still concise):**
        User Query: "Compare the average points per game for LeBron James and Stephen Curry over the last three seasons, and tell me which team had the best record in their conference last season."
        Output: `{{"sub_queries": [{{"query": "What were LeBron James' average points per game for the last three seasons?", "supervisor": "player_supervisor"}}, {{"query": "What were Stephen Curry's average points per game for the last three seasons?", "supervisor": "player_supervisor"}}, {{"query": "Which NBA team had the best record in their conference last season?", "supervisor": "teams_supervisor"}}]}}`

        **Example 5 (Advanced - Mixed Supervisors with Analysis):**
        User Query: "Analyze the impact of Draymond Green's defensive presence on the Golden State Warriors' win percentage over the past three seasons.  Also, what is his average blocks per game?"
        Output:  `{{"sub_queries": [{{"query": "What was the Golden State Warriors' win percentage over the past three seasons?", "supervisor": "teams_supervisor"}}, {{"query": "What is Draymond Green's average blocks per game over the past three seasons?", "supervisor": "player_supervisor"}}, {{"query": "How does Draymond Green's presence/absence correlate with the Warriors' win percentage over the past three seasons?", "supervisor": "teams_supervisor"}}]}}`

        **Example 6 (Advanced - Statistical Trends and Comparisons):**
        User Query: "Compare the assist-to-turnover ratio of Chris Paul, Rajon Rondo, and Russell Westbrook over their entire careers, and analyze how their passing styles have evolved."
        Output: `{{"sub_queries": [{{"query": "What is Chris Paul's career assist-to-turnover ratio?", "supervisor": "player_supervisor"}}, {{"query": "What is Rajon Rondo's career assist-to-turnover ratio?", "supervisor": "player_supervisor"}}, {{"query": "What is Russell Westbrook's career assist-to-turnover ratio?", "supervisor": "player_supervisor"}}, {{"query": "How has Chris Paul's passing style evolved over his career?", "supervisor": "player_supervisor"}}, {{"query": "How has Rajon Rondo's passing style evolved over his career?", "supervisor": "player_supervisor"}}, {{"query": "How has Russell Westbrook's passing style evolved over his career?", "supervisor": "player_supervisor"}}]}}`

        **Example 7 (Advanced - Game Strategy and Outcomes):**
        User Query: "Analyze the effectiveness of different defensive schemes (e.g., switching, hedging, dropping) against pick-and-roll plays involving Stephen Curry and Draymond Green.  Which scheme leads to the lowest points per possession for the opposing team, and how does this vary based on the personnel on the court?"
        Output: `{{"sub_queries": [{{"query": "What are the different defensive schemes used against pick-and-roll plays involving Stephen Curry and Draymond Green?", "supervisor": "game_supervisor"}}, {{"query": "What is the effectiveness (points per possession allowed) of switching defenses against Curry/Green pick-and-rolls?", "supervisor": "game_supervisor"}}, {{"query": "What is the effectiveness of hedging defenses against Curry/Green pick-and-rolls?", "supervisor": "game_supervisor"}}, {{"query": "What is the effectiveness of dropping defenses against Curry/Green pick-and-rolls?", "supervisor": "game_supervisor"}}, {{"query": "How does the effectiveness of different defensive schemes against Curry/Green pick-and-rolls vary based on opposing personnel?", "supervisor": "game_supervisor"}}]}}`
        
        """,
        input_variables=["query"]
    )

    chain = prompt | llm_with_structure
    query = state["messages"][-1].content
    structured_output = await chain.ainvoke({"query": query})
    return {"sub_queries": [sub_query.dict() for sub_query in structured_output.sub_queries]}  # Convert to dict



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



supervisor_dict = {
    "game_supervisor": game_supervisor,
    "player_supervisor": player_supervisor,
    "teams_supervisor": teams_supervisor,
}
workflow = StateGraph(AgentState)
workflow.add_node("split_query", split_node)
workflow.add_node("parallel_supervisors", lambda state, supervisors=supervisor_dict: asyncio.run(parallel_runner(state, supervisors))) # Changed to dict
workflow.add_node("combine_results", combine_results)

workflow.add_edge(START, "split_query")
workflow.add_edge("split_query", "parallel_supervisors")
workflow.add_edge("parallel_supervisors", "combine_results")
workflow.add_edge("combine_results", END)

app_nba = workflow.compile()



# initial_state = {
#     "messages": [HumanMessage(content="Tell me about the Rockets vs Pacers game happening today. What are the key stats to look out for?")]
# }
# final_state = app_nba.ainvoke(initial_state)
