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
from langchain.agents import create_react_agent, AgentExecutor
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
from app.react_agent.prompts import (SUPERVISOR_PROMPT, TEAM_PROMPT, PLAYER_PROMPT, GAME_INFO_PROMPT, GAME_DATA_PROMPT, 
                                     MAIN_SUPERVISOR_PROMPT, GAME_SUPERVISOR_PROMPT)
from app.react_agent.tools import (team_tools, player_tools,
                                   game_data_tools, game_info_tools)
# llm = LangchainChatDeepSeek(temperature=0, 
#                               streaming=True, 
#                               model="gpt-4-turbo",
#                               callbacks=AsyncCallbackManager([StreamingStdOutCallbackHandler()])
#             )


llm = LangchainChatDeepSeek(
                              model="gpt-4-turbo",
            )


# ---------------------------------------------------------------------
# 5. Node: "agent" 
# ---------------------------------------------------------------------

class SupervisorOutput(BaseModel):
    """
    The LLM must return JSON with these keys:
      - output: The final or partial answer to the user's question
      - is_satisfactory: True/False if we consider the answer complete
      - next_node: one of 'game_node', 'team_node', 'player_node', 'exit'
    """
    output: str = Field(..., description="The final or partial answer to the user's MLB-related query.")
    is_satisfactory: bool = Field(..., description="Whether the LLM considers the answer complete or not.")
    next_node: Literal["GameNode", "TeamNode", "PlayerNode", "exit"] = Field(
        ...,
        description="The next node to route to. 'exit' if no further nodes are needed."
    )
    
def supervisor_node(state: OverallState) -> Command[Literal["GameNode", "TeamNode", "PlayerNode", "__end__"]]:
    """
    This node:
      1. Uses an LLM that returns JSON with:
         { "output": "...", "satisfactory": bool, "next_node": game_node|team_node|player_node|exit" }
      2. If `satisfactory`=True or next_node='exit', we end the flow (goto=__end__).
      3. Otherwise, we route to next_node => "GameNode" or "TeamNode" or "PlayerNode".
      4. Append the 'output' from the LLM to state.messages as an AIMessage.
    """

    # If no user input, just end
    if not state.messages:
        return Command(goto="__end__")

    llm_with_structure = llm.with_structured_output(SupervisorOutput)

    # Define the prompt template
    prompt = PromptTemplate(
        template=SUPERVISOR_PROMPT,
        input_variables=["query"]
    )

    # Create the chain
    chain = prompt | llm_with_structure

    # Extract the user's query from state.messages
    query = state.messages[-1].content  # Assuming the last message is the user's query

    # Invoke the chain to generate the structured output
    structured_output = chain.invoke({"query": query})


    state.messages.append(AIMessage(content=structured_output.output))

    # If "satisfactory" is True OR next_node == "exit", we end
    if structured_output.is_satisfactory or structured_output.next_node.lower() == "exit":
        return Command(goto="__end__")
    else:
        # Map the next_node to the actual node name in your graph
        # We'll do a quick mapping:
        node_map = {
            "GameNode": "GameNode",
            "TeamNode": "TeamNode",
            "PlayerNode": "PlayerNode",
            "exit": "__end__"
        }
        next_node = node_map.get(structured_output.next_node, "__end__")
        return Command(goto=next_node)


# ---------------------------------------------------------------------------
# 3) Team Node (Has an advanced agent: LLM + Tools + Possibly a lookup mini-LLM)
# ---------------------------------------------------------------------------

def team_node(state: OverallState) -> OverallState:
    """
    This node has a 'React Agent' + multiple Tools (Team Tools, ID Lookup Tools).
    1) Build or call your advanced agent with React style + Tools.
    2) Possibly do an LLM-based lookup if the user only gave "Dodgers" name to get numeric ID.
    3) Then call the real 'Team Tools'.
    4) Append an AIMessage summarizing the outcome.
    """

    # Create the React agent prompt
    prompt = PromptTemplate.from_template(TEAM_PROMPT)
    
    # Create the React agent
    react_agent = create_react_agent(
        tools=team_tools,
        llm=llm,
        prompt=prompt,  # Using the MLB team prompt
    )
    
    agent_executor = AgentExecutor(
        agent=react_agent, 
        tools=team_tools, 
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    # Extract user input from the last message
    user_text = state.messages[0].content

    # Run the agent with the user query
    result = agent_executor.invoke({
        "input": f"Answer the questions about MLB teams, etc. using the available tools. {user_text}.",
        "query": user_text,
        "agent_scratchpad": ""  # Initialize with an empty scratchpad
    })
    
    try:
        # Your existing logic to generate `final_response`
        final_response = result.get("output") or result.get("Final Answer") or str(result)
    except Exception as e:
        # Handle errors and ensure `final_response` is a string
        final_response = f"An error occurred: {str(e)}"

    # Ensure `final_response` is a string or list
    if isinstance(final_response, dict):
        final_response = str(final_response)  # Convert dictionary to string

    return state



# ---------------------------------------------------------------------------
# 4) Player Node (Similarly advanced agent)
# ---------------------------------------------------------------------------
def player_node(state: OverallState) -> OverallState:
    """
    This node has a 'React Agent' + multiple Tools (Team Tools, ID Lookup Tools).
    1) Build or call your advanced agent with React style + Tools.
    2) Possibly do an LLM-based lookup if the user only gave "Dodgers" name to get numeric ID.
    3) Then call the real 'Team Tools'.
    4) Append an AIMessage summarizing the outcome.
    """

    # Create the React agent prompt
    prompt = PromptTemplate.from_template(PLAYER_PROMPT)
    
    # Create the React agent
    react_agent = create_react_agent(
        tools=player_tools,
        llm=llm,
        prompt=prompt,  # Using the MLB team prompt
    )
    
    agent_executor = AgentExecutor(
        agent=react_agent, 
        tools=player_tools, 
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    # Extract user input from the last message
    user_text = state.messages[0].content

    # Run the agent with the user query
    result = agent_executor.invoke({
        "input": f"Answer the questions about MLB players, stats, etc. using the available tools. {user_text}.",
        "query": user_text,
        "agent_scratchpad": ""  # Initialize with an empty scratchpad
    })
    
    try:
        # Your existing logic to generate `final_response`
        final_response = result.get("output") or result.get("Final Answer") or str(result)
    except Exception as e:
        # Handle errors and ensure `final_response` is a string
        final_response = f"An error occurred: {str(e)}"

    # Ensure `final_response` is a string or list
    if isinstance(final_response, dict):
        final_response = str(final_response)  # Convert dictionary to string

    return state




# ---------------------------------------------------------------------------
# 5) Game Node (Similarly advanced agent)
# ---------------------------------------------------------------------------
def game_node(state: OverallState) -> OverallState:
    """
    Another advanced agent specialized in game lookups.
    Possibly with a mini "lookup tool" for game_id, then calls mlb_get_live_game_data_tool, etc.
    """

    
    return state




# ---------------------------------------------------------------------------
# 6) Aggregator Node (Has an LLM to combine partial results)
# ---------------------------------------------------------------------------
def aggregator_node(state: OverallState) -> OverallState:
    """
    If user asked for multiple data (Team + Game + Player),
    we combine them into a final summary using a separate LLM if you want.
    """

    # # Combine partial data into a single prompt
    # summary_prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a final aggregator. Combine partial MLB data into a single summary."),
    #     ("user", f"TEAM data:\n{state['team_data']}\n\nGAME data:\n{state['game_data']}\n\nPLAYER data:\n{state['player_data']}"),
    # ])
    # aggregator_result = llm.invoke(prompt=summary_prompt)

    # state["aggregator_data"] = aggregator_result.content
    # state["messages"].append(AIMessage(content=f"(AGGREGATOR) {state['aggregator_data']}"))
    return state


# ----------------------------------------- MLB Node + SUpervisor Node -----------------------------------------
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
app = mlb_workflow.compile()


