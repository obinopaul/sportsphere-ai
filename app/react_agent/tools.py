"""This module provides example tools for for the LangChain platform.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast, Dict, Literal
from typing_extensions import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional, Dict, Any
from langchain.tools.base import StructuredTool
import os
from datetime import datetime

import re
from langchain.schema import HumanMessage, AIMessage
from langchain_core.messages import AnyMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.tools import BaseTool, Tool
import mlbstatsapi
import requests
import logging
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


#---------------------------------------------------------------------
from app.react_agent.configuration import Configuration
#---------------------------------------------------------------------

load_dotenv()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# If you're using something like LangChain Tools, uncomment or adjust the import:
# from langchain.tools import Tool

# -------------------------------------------------------------------
# 1) Get MLB 2024 (or any) Regular Season Schedule
# -------------------------------------------------------------------

class MLBGetScheduleInput(BaseModel):
    """
    Input schema for fetching MLB schedule data.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/schedule
    """
    sportId: Optional[int] = Field(1, description="Sport ID for MLB is 1.")
    season: Optional[str] = Field("2024", description="The season year. Example: '2024'.")
    gameType: Optional[str] = Field("R", description="Game type. Examples: R (Regular), P (Postseason), S (Spring).")
    date: Optional[str] = Field(
        None, 
        description="Specific date in MM/DD/YYYY format to get the schedule for that day."
    )
    # Add additional parameters as needed (e.g., fields, hydrate, etc.)

class MLBGetScheduleTool:
    """
    A tool to call the MLB StatsAPI /schedule endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/schedule"
        # No API key required for MLB endpoints.

    def run_get_schedule(
        self,
        sportId: int = 1,
        season: str = "2024",
        gameType: str = "R",
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        GET request to /schedule with optional query parameters.
        """
        params = {
            "sportId": sportId,
            "season": season,
            "gameType": gameType
        }
        if date:
            params["date"] = date

        try:
            resp = requests.get(self.base_url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_schedule_tool = StructuredTool(
    name="mlb_get_schedule",
    func=MLBGetScheduleTool().run_get_schedule,
    description="Calls the MLB StatsAPI to get the schedule for a given season, date, and game type.",
    args_schema=MLBGetScheduleInput
)

# -------------------------------------------------------------------
# 2) Get Team Roster
# -------------------------------------------------------------------

class MLBGetTeamRosterInput(BaseModel):
    """
    Input schema for fetching a specific team's roster.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/teams/{teamId}/roster
    """
    teamId: int = Field(..., description="Team ID. Example: 119 for LA Dodgers.")
    season: Optional[str] = Field(default = "2025", description="Season year. Example: '2024'.")

class MLBGetTeamRosterTool:
    """
    A tool to call the MLB StatsAPI /teams/{teamId}/roster endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/teams"

    def run_get_team_roster(self, teamId: int, season: str = "2024") -> Dict[str, Any]:
        """
        GET request to /teams/{teamId}/roster with optional season parameter.
        """
        url = f"{self.base_url}/{teamId}/roster"
        params = {
            "season": season
        }
        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_team_roster_tool = StructuredTool(
    name="mlb_get_team_roster",
    func=MLBGetTeamRosterTool().run_get_team_roster,
    description="Fetches a team's roster for a given season using the MLB StatsAPI.",
    args_schema=MLBGetTeamRosterInput
)

# -------------------------------------------------------------------
# 3) Get Team Information
# -------------------------------------------------------------------

class MLBGetTeamInfoInput(BaseModel):
    """
    Input schema for fetching detailed team info.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/teams/{teamId}
    """
    teamId: int = Field(..., description="Team ID. Example: 119 for LA Dodgers.")
    season: Optional[str] = Field(..., description="Season year. Example: '2024'.")

class MLBGetTeamInfoTool:
    """
    A tool to call the MLB StatsAPI /teams/{teamId} endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/teams"

    def run_get_team_info(self, teamId: int, season: Optional[str] = None) -> Dict[str, Any]:
        """
        GET request to /teams/{teamId} with optional season parameter.
        """
        url = f"{self.base_url}/{teamId}"
        params = {}
        if season:
            params["season"] = season

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_team_info_tool = StructuredTool(
    name="mlb_get_team_info",
    func=MLBGetTeamInfoTool().run_get_team_info,
    description="Fetches detailed information about a given MLB team from the StatsAPI.",
    args_schema=MLBGetTeamInfoInput
)

# -------------------------------------------------------------------
# 4) Get Player Information
# -------------------------------------------------------------------

class MLBGetPlayerInfoInput(BaseModel):
    """
    Input schema for fetching a specific player's info.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1/people/{playerId}
    """
    playerId: int = Field(..., description="Player ID. Example: 660271 for Shohei Ohtani.")
    season: Optional[str] = Field(..., description="Season year. Example: '2024'.")

class MLBGetPlayerInfoTool:
    """
    A tool to call the MLB StatsAPI /people/{playerId} endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1/people"

    def run_get_player_info(self, playerId: int, season: Optional[str] = None) -> Dict[str, Any]:
        """
        GET request to /people/{playerId} with optional season parameter.
        """
        url = f"{self.base_url}/{playerId}"
        params = {}
        if season:
            params["season"] = season

        try:
            resp = requests.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_player_info_tool = StructuredTool(
    name="mlb_get_player_info",
    func=MLBGetPlayerInfoTool().run_get_player_info,
    description="Fetches detailed information about a specific MLB player.",
    args_schema=MLBGetPlayerInfoInput
)

# -------------------------------------------------------------------
# 5) Get Live Game Data (GUMBO Feed)
# -------------------------------------------------------------------

class MLBGetLiveGameDataInput(BaseModel):
    """
    Input schema for fetching GUMBO live feed (entire game state).
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live
    """
    game_pk: int = Field(..., description="Game primary key (e.g., 716463).")

class MLBGetLiveGameDataTool:
    """
    A tool to call the MLB StatsAPI /game/{game_pk}/feed/live endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def run_get_live_game_data(self, game_pk: int) -> Dict[str, Any]:
        """
        GET request to /game/{game_pk}/feed/live to get the GUMBO feed for a specific game.
        """
        url = f"{self.base_url}/{game_pk}/feed/live"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_live_game_data_tool = StructuredTool(
    name="mlb_get_live_game_data",
    func=MLBGetLiveGameDataTool().run_get_live_game_data,
    description="Fetches the GUMBO live feed for a specified MLB game.",
    args_schema=MLBGetLiveGameDataInput
)

# -------------------------------------------------------------------
# 6) Get Game Timestamps
# -------------------------------------------------------------------

class MLBGetGameTimestampsInput(BaseModel):
    """
    Input schema for fetching the timestamps of GUMBO updates for a given game.
    Uses the StatsAPI endpoint: https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live/timestamps
    """
    game_pk: int = Field(..., description="Game primary key (e.g., 716463).")

class MLBGetGameTimestampsTool:
    """
    A tool to call the MLB StatsAPI /game/{game_pk}/feed/live/timestamps endpoint.
    """
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1.1/game"

    def run_get_game_timestamps(self, game_pk: int) -> Dict[str, Any]:
        """
        GET request to /game/{game_pk}/feed/live/timestamps for update timestamps.
        """
        url = f"{self.base_url}/{game_pk}/feed/live/timestamps"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"RequestException: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

mlb_get_game_timestamps_tool = StructuredTool(
    name="mlb_get_game_timestamps",
    func=MLBGetGameTimestampsTool().run_get_game_timestamps,
    description="Fetches the list of GUMBO update timestamps for a given MLB game.",
    args_schema=MLBGetGameTimestampsInput
)

# game_data_tools = [mlb_get_schedule_tool, mlb_get_live_game_data_tool, mlb_get_game_timestamps_tool]
# team_tools = [mlb_get_team_roster_tool, mlb_get_team_info_tool]
# player_tools = [mlb_get_player_info_tool]

# -------------------------------------------------------------------
# End of Tools
# -------------------------------------------------------------------

# You now have six robust tools for common MLB StatsAPI queries:
# 1) mlb_get_schedule_tool
# 2) mlb_get_team_roster_tool
# 3) mlb_get_team_info_tool
# 4) mlb_get_player_info_tool
# 5) mlb_get_live_game_data_tool
# 6) mlb_get_game_timestamps_tool

# You can import and use them as needed in your project. 
# For example:
# result = mlb_get_schedule_tool.func(sportId=1, season="2024", gameType="R", date="03/28/2024")
# print(result)


# -------------------------------------------------------------------
# 7) Get Team ID From Team Name
# -------------------------------------------------------------------

class MLBGetTeamIdInput(BaseModel):
    """
    Input schema for retrieving MLB team ID(s) by a team name string.
    This uses mlb.get_team_id(team_name, search_key=...) under the hood.
    """
    team_name: str = Field(..., description="Full or partial team name, e.g. 'Oakland Athletics'.")
    search_key: Optional[str] = Field(
        "name",
        description="Which search field to match on; defaults to 'name'."
    )


class MLBGetTeamIdTool:
    """
    A tool that calls python-mlb-statsapi's Mlb.get_team_id().
    Returns a list of matching team IDs.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_team_id(self, team_name: str, search_key: str = "name") -> Dict[str, Any]:
        """
        Returns: A dict with the list of matching team IDs and a success/error message.
        """
        try:
            team_ids = self.client.get_team_id(team_name, search_key=search_key)
            return {
                "team_name": team_name,
                "matching_team_ids": team_ids
            }
        except Exception as e:
            return {"error": f"Unable to retrieve team ID(s): {str(e)}"}


mlb_get_team_id_tool = StructuredTool(
    name="mlb_get_team_id",
    func=MLBGetTeamIdTool().run_get_team_id,
    description="Get a list of MLB team ID(s) by providing a team name.",
    args_schema=MLBGetTeamIdInput
)



# -------------------------------------------------------------------
# 8) Get Player ID From Full Name
# -------------------------------------------------------------------

class MLBGetPlayerIdInput(BaseModel):
    """
    Input schema for retrieving MLB player ID(s) by a player name string.
    """
    player_name: str = Field(..., description="Player's name, e.g. 'Shohei Ohtani' or 'Ty France'.")
    sport_id: Optional[int] = Field(default = 1, description="Sport ID, defaults to 1 for MLB.")
    search_key: Optional[str] = Field(
        default = "fullname",
        description="Which search field to match on; typically 'fullname'."
    )


class MLBGetPlayerIdTool:
    """
    A tool that calls python-mlb-statsapi's Mlb.get_people_id().
    Returns a list of matching player IDs.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_player_id(
        self,
        player_name: str,
        sport_id: int = 1,
        search_key: str = "fullname"
    ) -> Dict[str, Any]:
        try:
            player_ids = self.client.get_people_id(
                fullname=player_name,
                sport_id=sport_id,
                search_key=search_key
            )
            # return {
            #     "player_name": player_name,
            #     "matching_player_ids": player_ids
            # }
    
            if player_ids:
                return f"Player: {player_name}, Matching Player IDs: {', '.join(map(str, player_ids))}"
            else:
                return f"No matching player IDs found for: {player_name}"
            
        except Exception as e:
            return {"error": f"Unable to retrieve player ID(s): {str(e)}"}


mlb_get_player_id_tool = StructuredTool(
    name="mlb_get_player_id",
    func=MLBGetPlayerIdTool().run_get_player_id,
    description="Get a list of MLB player IDs by providing a full player name.",
    args_schema=MLBGetPlayerIdInput
)



from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import mlbstatsapi

    
# -------------------------------------------------------------------
# 10) Get Game PK (IDs) By Date
# -------------------------------------------------------------------


class MLBGetGameIdsByDateInput(BaseModel):
    """
    Input schema to retrieve a list of game_pk IDs for a given date.
    """
    date: str = Field(..., description="Date in YYYY-MM-DD format.")
    sport_id: Optional[int] = Field(default = 1, description="Sport ID for MLB is 1.")
    team_id: Optional[int] = Field(..., description="Filter by a specific team's ID if desired.")


class MLBGetGameIdsByDateTool:
    """
    A tool that calls python-mlb-statsapi's Mlb.get_scheduled_games_by_date().
    Returns a list of game IDs for the given date (and optional team).
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_game_ids_by_date(
        self,
        date: str,
        sport_id: int = 1,
        team_id: Optional[int] = None
    ) -> Dict[str, Any]:
        try:
            # This method returns a list of game IDs. If no games found, might be an empty list.
            game_ids = self.client.get_scheduled_games_by_date(
                date=date,
                sport_id=sport_id,
                team_id=team_id
            )
            return {
                "requested_date": date,
                "sport_id": sport_id,
                "team_id": team_id,
                "game_ids": game_ids
            }
        except Exception as e:
            return {"error": f"Unable to retrieve game IDs: {str(e)}"}


mlb_get_game_ids_by_date_tool = StructuredTool(
    name="mlb_get_game_ids_by_date",
    func=MLBGetGameIdsByDateTool().run_get_game_ids_by_date,
    description="Get a list of MLB game_pk (IDs) scheduled on a specific date using python-mlb-statsapi.",
    args_schema=MLBGetGameIdsByDateInput
)




# -------------------------------------------------------------------
# 11) Get a Single Game’s “PK” by Searching Team & Date
# -------------------------------------------------------------------

class MLBFindOneGameIdInput(BaseModel):
    """
    Input schema to find the first game PK matching a team on a certain date.
    """
    date: str = Field(..., description="Date in YYYY-MM-DD format.")
    team_name: str = Field(..., description="Team name, e.g. 'Seattle Mariners'.")


class MLBFindOneGameIdTool:
    """
    A tool that:
      1) Gets the team_id from the name (using get_team_id).
      2) Then calls get_scheduled_games_by_date(date=..., team_id=TEAM_ID).
      3) Returns the first found game_pk or all of them if you prefer.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_find_one_game_id(self, date: str, team_name: str) -> Dict[str, Any]:
        try:
            # 1) Find the team_id
            team_ids = self.client.get_team_id(team_name)
            if not team_ids:
                return {"error": f"No team ID found for '{team_name}'."}
            team_id = team_ids[0]

            # 2) Grab the game IDs for that date/team
            game_ids = self.client.get_scheduled_games_by_date(
                date=date,
                sport_id=1,
                team_id=team_id
            )

            if not game_ids:
                return {
                    "date": date,
                    "team_name": team_name,
                    "error": "No games found for this date/team."
                }

            # For demonstration: just return the first game
            return {
                "date": date,
                "team_id": team_id,
                "found_game_ids": game_ids,
                "first_game_id": game_ids[0]
            }

        except Exception as e:
            return {"error": f"Unable to find game ID: {str(e)}"}


mlb_find_one_game_id_tool = StructuredTool(
    name="mlb_find_one_game_id",
    func=MLBFindOneGameIdTool().run_find_one_game_id,
    description="Search for the first MLB game_pk on a given date for a given team name.",
    args_schema=MLBFindOneGameIdInput
)




# -------------------------------------------------------------------
# 12) Get Venue ID By Name
# -------------------------------------------------------------------

class MLBGetVenueIdInput(BaseModel):
    venue_name: str = Field(..., description="Venue name, e.g. 'PNC Park' or 'Wrigley Field'.")
    search_key: Optional[str] = Field(default = "name", description="Search field to match on.")


class MLBGetVenueIdTool:
    """
    A tool to call Mlb.get_venue_id(...), returning a list of matching venue IDs.
    """
    def __init__(self):
        self.client = mlbstatsapi.Mlb()

    def run_get_venue_id(self, venue_name: str, search_key: str = "name") -> Dict[str, Any]:
        try:
            venue_ids = self.client.get_venue_id(venue_name, search_key=search_key)
            return {
                "venue_name": venue_name,
                "matching_venue_ids": venue_ids
            }
        except Exception as e:
            return {"error": f"Unable to retrieve venue ID(s): {str(e)}"}


mlb_get_venue_id_tool = StructuredTool(
    name="mlb_get_venue_id",
    func=MLBGetVenueIdTool().run_get_venue_id,
    description="Get a list of venue IDs for a stadium name (e.g. 'Wrigley Field').",
    args_schema=MLBGetVenueIdInput
)



# -------------------------------------------------------------------
# 13) Tavily Search Tool
# -------------------------------------------------------------------
# Define Input Schema# Define Input Schema
class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up.")
    max_results: Optional[int] = Field(default=10, description="The maximum number of search results to return.")

# Define the Tool
class TavilySearchTool:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform a web search using the Tavily search engine.
        """
        try:
            # Initialize the Tavily search tool with the configured max_results
            search_tool = TavilySearchResults(max_results=self.max_results)

            # Perform the search
            result = search_tool.invoke({"query": query})

            # Return the search results
            return result
        except Exception as e:
            return {"error": str(e)}

# Create the LangChain Tool
tavily_search_tool = StructuredTool(
    name="tavily_search",
    func=TavilySearchTool().search,
    description="Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.",
    args_schema=SearchToolInput
)


# -------------------------------------------------------------------
# ID Lookup Tools
# -------------------------------------------------------------------
# team_id_lookup_tools = [mlb_get_team_id_tool], 
team_tools = [mlb_get_team_id_tool, mlb_get_team_roster_tool, mlb_get_team_info_tool]
player_tools = [mlb_get_player_id_tool, mlb_get_player_info_tool, tavily_search_tool]

# game_id_lookup_tools = [mlb_get_game_ids_by_date_tool, mlb_find_one_game_id_tool, mlb_get_venue_id_tool, tavily_search_tool]
# game_data_tools = [mlb_get_game_ids_by_date_tool, mlb_get_schedule_tool, mlb_get_live_game_data_tool, mlb_get_game_timestamps_tool, tavily_search_tool]

game_info_tools = [mlb_get_game_ids_by_date_tool, mlb_find_one_game_id_tool, tavily_search_tool]
game_data_tools = [mlb_get_game_ids_by_date_tool, mlb_get_schedule_tool, mlb_get_live_game_data_tool]

