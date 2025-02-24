from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict, Union, Literal
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
import os

# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define the output class for structured output
class StructuredOutput(BaseModel):
    param1: str
    param2: str

class Assistant:
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable
    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)
            
            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break
        # Return the final state after processing the runnable
        return {"messages": result}
    
class StructuredReactAgent:
    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list,
        prompt: ChatPromptTemplate,
        output_class: BaseModel = None,
    ):
        """
        Initialize the StructuredReactAgent.

        Args:
            llm: The language model to use.
            tools: A list of tools available to the agent.
            prompt: The prompt template for the agent.
            output_class: The Pydantic model for structured output (optional).
        """
        self.llm = llm
        self.tools = tools
        self.prompt = prompt
        self.output_class = output_class

        # Bind tools to the LLM
        self.assistant_runnable = self.prompt | self.llm.bind_tools(self.tools, tool_choice="any")

    def handle_tool_error(self, state) -> dict:
        """
        Handle errors that occur during tool execution.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing error messages for each tool that encountered an issue.
        """
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    def create_tool_node_with_fallback(self) -> dict:
        """
        Create a tool node with fallback error handling.

        Returns:
            A tool node that uses fallback behavior in case of errors.
        """
        return ToolNode(self.tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)],
            exception_key="error",
        )

    def tools_condition(self, state: Union[list[AnyMessage], dict[str, Any], BaseModel]) -> Literal["tools", "review"]:
        """
        Determine whether to route to the ToolNode or the review node.

        Args:
            state: The current state of the agent.

        Returns:
            The next node to route to ("tools" or "review").
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif isinstance(state, dict) and (messages := state.get("messages", [])):
            ai_message = messages[-1]
        elif messages := getattr(state, "messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "review"

    def review_model(self, state: MessagesState):
        """
        Generate structured output using the review node.

        Args:
            state: The current state of the agent.

        Returns:
            A dictionary containing the structured output.
        """
        messages = state["messages"]
        if self.output_class:
            return {"messages": [self.llm.with_structured_output(self.output_class).invoke(messages)]}
        return {"messages": messages}  # Fallback if no output class is provided

    def build_graph(self):
        """
        Build the StateGraph for the agent.

        Returns:
            A compiled StateGraph.
        """
        builder = StateGraph(State)

        # Add nodes
        builder.add_node("assistant", Assistant(self.assistant_runnable))
        builder.add_node("tools", self.create_tool_node_with_fallback())
        builder.add_node("review", self.review_model)

        # Add edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", self.tools_condition)
        builder.add_edge("tools", "assistant")
        builder.add_edge("review", END)

        # Compile the graph
        memory = MemorySaver()
        return builder.compile()

    def display_graph(self):
        """
        Display the graph for the agent.
        """
        from IPython.display import Image, display
        
        graph = self.build_graph()
        display(Image(graph.get_graph().draw_mermaid_png()))



# Example usage
if __name__ == "__main__":
    # Initialize the LLM
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        streaming=True,
        callbacks=AsyncCallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # Define the tools
    tools = [TavilySearchResults()]  # Add your tools here

    # Define the prompt
    primary_assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are a helpful customer support assistant for Solar Panels Belgium.
                You should get the following information from them:
                - monthly electricity cost
                If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
                After you are able to discern all the information, call the relevant tool.
                ''',
            ),
            ("placeholder", "{messages}"),
        ]
    )

    # Initialize the agent
    agent = StructuredReactAgent(
        llm=llm,
        tools=tools,
        prompt=primary_assistant_prompt,
        output_class=StructuredOutput,  # Optional: Define your output class
    )

    # Build and run the graph
    graph = agent.build_graph()
    initial_state = {"messages": [("user", "Tell me about solar panels in Belgium.")]}
    result = graph.invoke(initial_state)
    print(result)

