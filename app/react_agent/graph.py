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




# ---------------------------------------------------------------------
# 1) Graph
# ---------------------------------------------------------------------
workflow = StateGraph(OverallState)


# ---------------------------------------------------------------------
# 2) Nodes
# ---------------------------------------------------------------------
# workflow.add_node("agent", agent_node)
# # workflow.add_node("retrieve", retrieve_node)
# workflow.add_node("retrieve", create_tool_node_with_fallback([retriever_tool]))

# # ---------------------------------------------------------------------
# # 3) Edges
# # ---------------------------------------------------------------------
# workflow.add_edge(START, "agent")
# workflow.add_conditional_edges(
#     "agent",
#     tools_condition,  # decides if the agent is calling a tool or finishing
#     {
#         "tools": "retrieve",
#         END: END,  # Ensure the agent can terminate
#     },
# )

# # If we generate, we go back to the agent
# workflow.add_edge("retrieve", "agent")


#---------------- Nodes ----------------
workflow.add_node("SupervisorNode", supervisor_node)
workflow.add_node("TeamNode", team_node)
workflow.add_node("GameNode", game_node)
workflow.add_node("PlayerNode", player_node)
workflow.add_node("AggregatorNode", aggregator_node)


#---------------- Edges ----------------
workflow.add_edge(START, "SupervisorNode")


# After sub-nodes, we typically return control to SupervisorNode or go aggregator
workflow.add_edge("GameNode", "AggregatorNode")
workflow.add_edge("PlayerNode", "AggregatorNode")
workflow.add_edge("TeamNode", "AggregatorNode")

# aggregator -> end
workflow.add_edge("AggregatorNode", END)



# ---------------------------------------------------------------------
# 4) Compile the graph
# ---------------------------------------------------------------------
graph = workflow.compile()
graph.name = "Reasoning and Action Agent"












# # ----------------------------------------- with Audio -----------------------------------------
# import io
# import threading
# import numpy as np
# import sounddevice as sd
# from scipy.io.wavfile import write
# from IPython.display import Image, display

# from openai import OpenAI

# from elevenlabs import play, VoiceSettings
# from elevenlabs.client import ElevenLabs

# from langgraph.graph import StateGraph, MessagesState, END, START

# # Initialize OpenAI client
# openai_client = OpenAI()

# # Initialize ElevenLabs client
# elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# def record_audio_until_stop(state: MessagesState):

#     """Records audio from the microphone until Enter is pressed, then saves it to a .wav file."""
    
#     audio_data = []  # List to store audio chunks
#     recording = True  # Flag to control recording
#     sample_rate = 16000 # (kHz) Adequate for human voice frequency

#     def record_audio():
#         """Continuously records audio until the recording flag is set to False."""
#         nonlocal audio_data, recording
#         with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
#             print("Recording your instruction! ... Press Enter to stop recording.")
#             while recording:
#                 audio_chunk, _ = stream.read(1024)  # Read audio data in chunks
#                 audio_data.append(audio_chunk)

#     def stop_recording():
#         """Waits for user input to stop the recording."""
#         input()  # Wait for Enter key press
#         nonlocal recording
#         recording = False

#     # Start recording in a separate thread
#     recording_thread = threading.Thread(target=record_audio)
#     recording_thread.start()

#     # Start a thread to listen for the Enter key
#     stop_thread = threading.Thread(target=stop_recording)
#     stop_thread.start()

#     # Wait for both threads to complete
#     stop_thread.join()
#     recording_thread.join()

#     # Stack all audio chunks into a single NumPy array and write to file
#     audio_data = np.concatenate(audio_data, axis=0)
    
#     # Convert to WAV format in-memory
#     audio_bytes = io.BytesIO()
#     write(audio_bytes, sample_rate, audio_data)  # Use scipy's write function to save to BytesIO
#     audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
#     audio_bytes.name = "audio.wav" # Set a filename for the in-memory file

#     # Transcribe via Whisper
#     transcription = openai_client.audio.transcriptions.create(
#        model="whisper-1", 
#        file=audio_bytes,
#     )

#     # Print the transcription
#     print("Here is the transcription:", transcription.text)

#     # Write to messages 
#     return {"messages": [HumanMessage(content=transcription.text)]}

# def play_audio(state: MessagesState):
    
#     """Plays the audio response from the remote graph with ElevenLabs."""

#     # Response from the agent 
#     response = state['messages'][-1]

#     # Prepare text by replacing ** with empty strings
#     # These can cause unexpected behavior in ElevenLabs
#     cleaned_text = response.content.replace("**", "")
    
#     # Call text_to_speech API with turbo model for low latency
#     response = elevenlabs_client.text_to_speech.convert(
#         voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
#         output_format="mp3_22050_32",
#         text=cleaned_text,
#         model_id="eleven_turbo_v2_5", 
#         voice_settings=VoiceSettings(
#             stability=0.0,
#             similarity_boost=1.0,
#             style=0.0,
#             use_speaker_boost=True,
#         ),
#     )
    
#     # Play the audio back
#     play(response)

# # Define parent graph
# builder = StateGraph(MessagesState)

# # Add remote graph directly as a node
# builder.add_node("audio_input", record_audio_until_stop)
# builder.add_node("todo_app", remote_graph)
# builder.add_node("audio_output", play_audio)
# builder.add_edge(START, "audio_input")
# builder.add_edge("audio_input", "todo_app")
# builder.add_edge("todo_app","audio_output")
# builder.add_edge("audio_output",END)
# graph = builder.compile()

# display(Image(graph.get_graph(xray=1).draw_mermaid_png()))