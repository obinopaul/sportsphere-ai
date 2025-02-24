"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Literal

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Sequence, Annotated
from langchain.schema import BaseMessage
from typing_extensions import TypedDict
import operator
from datetime import date


@dataclass
class OverallState:
    
    # Agent Workflow Tracking
    messages: Annotated[list[AnyMessage], add_messages]
    "Stores the sequence of messages exchanged between the user and the agent."

