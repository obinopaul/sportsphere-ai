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




# ---------------------------------------------------- NBA TOOLS ------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------
# 1) ScoreBoard Tool (Live Endpoint)
# -------------------------------------------------------------------
# Retrieves today’s scoreboard data from the live endpoint.

from langchain.tools import StructuredTool
from nba_api.live.nba.endpoints import scoreboard

# ========== 1) Define Input Schema ==========
class LiveScoreBoardInput(BaseModel):
    """
    Schema for fetching the current day scoreboard (live games).
    No extra parameters for scoreboard, but you can add filters if needed.
    """
    dummy_param: Optional[str] = Field(
        default="",
        description="Not used, but placeholder for expansions if needed."
    )

# ========== 2) Define the Tool Class ==========
class NBAFetchScoreBoardTool:
    """
    Fetch today's scoreboard from the NBA Live endpoint.
    """
    def __init__(self):
        pass  # Any initial config can go here if needed

    def run(self, dummy_param: Optional[str] = "") -> Dict[str, Any]:
        """
        Gets the scoreboard data for today's NBA games.
        Returns it as a dictionary.
        """
        try:
            sb = scoreboard.ScoreBoard()  # Instantiate scoreboard
            data_dict = sb.get_dict()     # Dictionary of scoreboard data
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
live_scoreboard_tool = StructuredTool(
    name="nba_live_scoreboard",
    description=(
        "Fetch today's NBA scoreboard (live or latest). "
        "Useful for retrieving the current day's games, scores, period, status, etc."
    ),
    func=NBAFetchScoreBoardTool().run,
    args_schema=LiveScoreBoardInput
)


# -------------------------------------------------------------------
# 2) BoxScore Tool (Live Endpoint)
# -------------------------------------------------------------------
# Given a valid NBA game_id, retrieve the real-time box score from the live endpoint.

from nba_api.live.nba.endpoints import boxscore

# ========== 1) Define Input Schema ==========
class LiveBoxScoreInput(BaseModel):
    """
    Schema for fetching box score data using live/nba/endpoints/boxscore.
    """
    game_id: str = Field(
        ...,
        description="A 10-digit NBA game ID (e.g., '0022200017')."
    )

# ========== 2) Define the Tool Class ==========
class NBAFetchBoxScoreTool:
    """
    Fetches a real-time box score for a given game ID from NBA Live endpoints.
    """
    def __init__(self):
        pass

    def run(self, game_id: str) -> Dict[str, Any]:
        """
        Return the box score as a dictionary.
        """
        try:
            bs = boxscore.BoxScore(game_id=game_id)
            data_dict = bs.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
live_boxscore_tool = StructuredTool(
    name="nba_live_boxscore",
    description=(
        "Fetch the real-time (live) box score for a given NBA game ID. "
        "Provides scoring, stats, team info, and player data."
    ),
    func=NBAFetchBoxScoreTool().run,
    args_schema=LiveBoxScoreInput
)



# -------------------------------------------------------------------
# 3) PlayByPlay Tool (Live Endpoint)
# -------------------------------------------------------------------
# Pulls the real-time play-by-play feed for a given game_id.

from nba_api.live.nba.endpoints import playbyplay

# ========== 1) Define Input Schema ==========
class LivePlayByPlayInput(BaseModel):
    """
    Schema for live PlayByPlay data retrieval.
    """
    game_id: str = Field(
        ...,
        description="A 10-digit NBA game ID for which to fetch play-by-play actions."
    )

# ========== 2) Define the Tool Class ==========
class NBAFetchPlayByPlayTool:
    """
    Fetch real-time play-by-play data from the NBA Live endpoint for the given game ID.
    """
    def __init__(self):
        pass

    def run(self, game_id: str) -> Dict[str, Any]:
        """
        Return the play-by-play feed as a dictionary.
        """
        try:
            pbp = playbyplay.PlayByPlay(game_id=game_id)
            data_dict = pbp.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
live_playbyplay_tool = StructuredTool(
    name="nba_live_play_by_play",
    description=(
        "Retrieve the live play-by-play actions for a specific NBA game ID. "
        "Useful for real-time game event tracking."
    ),
    func=NBAFetchPlayByPlayTool().run,
    args_schema=LivePlayByPlayInput
)

# -------------------------------------------------------------------
# 4) CommonPlayerInfo Tool (Stats Endpoint)
# -------------------------------------------------------------------
# Retrieve standard information about an NBA player (e.g., birthdate, height, years of experience).
from nba_api.stats.endpoints import commonplayerinfo

# ========== 1) Define Input Schema ==========
class CommonPlayerInfoInput(BaseModel):
    """
    Pydantic schema for requesting common player info from stats.nba.com
    """
    player_id: str = Field(
        ...,
        description="NBA player ID (e.g., '2544' for LeBron James)."
    )


# ========== 2) Define the Tool Class ==========
class NBACommonPlayerInfoTool:
    """
    Retrieve player's basic profile data from the stats.nba.com endpoint.
    """
    def __init__(self):
        pass

    def run(self, player_id: str, league_id: str = "00") -> Dict[str, Any]:
        """
        Return data as dictionary, including personal info, stats, etc.
        """
        try:
            info = commonplayerinfo.CommonPlayerInfo(
                player_id=player_id
            )
            data_dict = info.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
common_player_info_tool = StructuredTool(
    name="nba_common_player_info",
    description=(
        "Retrieve basic information about a player (height, weight, birthdate, "
        "team, experience, etc.) from NBA stats endpoints."
    ),
    func=NBACommonPlayerInfoTool().run,
    args_schema=CommonPlayerInfoInput
)


# -------------------------------------------------------------------
# 5) PlayerCareerStats Tool (Stats Endpoint)
# -------------------------------------------------------------------
# Retrieves career stats for a given player (split by season and possibly by team).

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from nba_api.stats.endpoints import playercareerstats

# ========== 1) Define Input Schema ==========
class PlayerCareerStatsInput(BaseModel):
    """
    Schema for retrieving a player's aggregated career stats.
    """
    player_id: str = Field(
        ...,
        description="NBA player ID (e.g., '203999' for Nikola Jokic)."
    )
    per_mode: Optional[str] = Field(
        default="PerGame",
        description="One of 'Totals', 'PerGame', 'Per36', etc."
    )

# ========== 2) Define the Tool Class ==========
class NBAPlayerCareerStatsTool:
    """
    Pull aggregated career stats (regular season & playoff) for an NBA player from stats.nba.com.
    """
    def __init__(self):
        pass

    def run(self, player_id: str, per_mode: str = "PerGame") -> Dict[str, Any]:
        """
        Returns a dictionary containing the player's career data.
        """
        try:
            career = playercareerstats.PlayerCareerStats(
                player_id=player_id,
                per_mode36=per_mode  # param name is per_mode36 in the library
            )
            data_dict = career.get_dict()
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
player_career_stats_tool = StructuredTool(
    name="nba_player_career_stats",
    description=(
        "Obtain an NBA player's career statistics (regular season, playoffs, etc.) "
        "from the stats.nba.com endpoints. Usage requires a valid player_id."
    ),
    func=NBAPlayerCareerStatsTool().run,
    args_schema=PlayerCareerStatsInput
)


# -------------------------------------------------------------------
# 6) Search Players by Name
# -------------------------------------------------------------------
from nba_api.stats.static import players

# ========== 1) Define Input Schema ==========
class SearchPlayersByNameInput(BaseModel):
    name_query: str = Field(
        ...,
        description="Full or partial name of the player to look up (e.g. 'LeBron', 'Curry', 'James')."
    )

# ========== 2) Define the Tool Class ==========
class NBAPlayerSearchTool:
    """
    Searches NBA players by name (case-insensitive) using the static library in nba_api.
    Returns a list of matches with IDs, full names, etc.
    """
    def __init__(self):
        pass

    def run(self, name_query: str) -> List[Dict[str, Any]]:
        """
        Returns a list of player dicts: 
        [
          {
            'id': <player_id>,
            'full_name': 'FirstName LastName',
            'first_name': ...,
            'last_name': ...,
            'is_active': ...
          }, 
          ...
        ]
        """
        try:
            results = players.find_players_by_full_name(name_query)
            return results
        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
search_players_by_name_tool = StructuredTool(
    name="nba_search_players",
    description=(
        "Search NBA players by partial or full name. "
        "Returns a list of matches with 'id' fields which can be used as 'player_id'."
    ),
    func=NBAPlayerSearchTool().run,
    args_schema=SearchPlayersByNameInput
)


# -------------------------------------------------------------------
# 7) Search Teams by Name
# -------------------------------------------------------------------
from nba_api.stats.static import teams

# ========== 1) Define Input Schema ==========
class SearchTeamsByNameInput(BaseModel):
    name_query: str = Field(
        ...,
        description="Full or partial team name (e.g. 'Lakers', 'Cavaliers')."
    )

# ========== 2) Define the Tool Class ==========
class NBATeamSearchTool:
    """
    Searches NBA teams by partial or full name using the static library in nba_api.
    """
    def __init__(self):
        pass

    def run(self, name_query: str) -> List[Dict[str, Any]]:
        """
        Returns a list of team dicts:
        [
          {
            'id': <team_id>,
            'full_name': 'Los Angeles Lakers',
            'abbreviation': 'LAL',
            'nickname': 'Lakers',
            'city': 'Los Angeles',
            'state': 'California',
            'year_founded': 1948
          },
          ...
        ]
        """
        try:
            results = teams.find_teams_by_full_name(name_query)
            return results
        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
search_teams_by_name_tool = StructuredTool(
    name="nba_search_teams",
    description=(
        "Search NBA teams by partial or full name. "
        "Returns a list of matches with 'id' used as 'team_id'."
    ),
    func=NBATeamSearchTool().run,
    args_schema=SearchTeamsByNameInput
)


# -------------------------------------------------------------------
# 8) List All Active Players
# -------------------------------------------------------------------
from nba_api.stats.static import players

# ========== 1) Define Input Schema ==========
class ListActivePlayersInput(BaseModel):
    # no arguments needed
    dummy: str = "unused"

# ========== 2) Define the Tool Class ==========
class NBAListActivePlayersTool:
    """
    Lists all active NBA players as a big dictionary list, 
    each containing 'id', 'full_name', 'is_active', etc.
    """
    def __init__(self):
        pass

    def run(self, dummy: str = "") -> List[Dict[str, Any]]:
        try:
            all_active = players.get_active_players()
            return all_active
        except Exception as e:
            return [{"error": str(e)}]

# ========== 3) Create the LangChain StructuredTool ==========
list_active_players_tool = StructuredTool(
    name="nba_list_active_players",
    description=(
        "Return a list of all currently active NBA players with their IDs and names. "
        "No input needed."
    ),
    func=NBAListActivePlayersTool().run,
    args_schema=ListActivePlayersInput
)


# -------------------------------------------------------------------
# 9) List Today’s Games (Stats vs. Live)
# -------------------------------------------------------------------
from nba_api.stats.endpoints import scoreboardv2

# ========== 1) Define Input Schema ==========
class TodayGamesInput(BaseModel):
    game_date: str = Field(
        ...,
        description="A date in 'YYYY-MM-DD' format to fetch scheduled or completed games."
    )

    league_id: str = Field(
        default="00",
        description="League ID (default=00 for NBA)."
    )

# ========== 2) Define the Tool Class ==========
class NBATodayGamesTool:
    """
    Lists the scoreboard from stats.nba.com for a given date, returning the games data set.
    """
    def __init__(self):
        pass

    def run(self, game_date: str, league_id: str = "00") -> Dict[str, Any]:
        """
        Return scoreboard details as a dictionary. 
        Typically you can find 'GAME_ID' in the 'GameHeader' dataset.
        """
        try:
            sb = scoreboardv2.ScoreboardV2(
                game_date=game_date,
                league_id=league_id,
            )
            data_dict = sb.get_normalized_dict()  # or .get_dict() if you prefer raw structure
            return data_dict
        except Exception as e:
            return {"error": str(e)}

# ========== 3) Create the LangChain StructuredTool ==========
list_todays_games_tool = StructuredTool(
    name="nba_list_todays_games",
    description=(
        "Returns scoreboard data from stats.nba.com for a given date (YYYY-MM-DD), "
        "including the game IDs, matchups, status, etc."
    ),
    func=NBATodayGamesTool().run,
    args_schema=TodayGamesInput
)



# -------------------------------------------------------------------
# 10) TeamGameLogsTool: Fetch a Team's Game Logs
# -------------------------------------------------------------------
from nba_api.stats.endpoints import teamgamelogs

# 1) Define Input Schema
class TeamGameLogsInput(BaseModel):
    """
    Tool input for fetching a team's game logs (and thus their game IDs).
    """
    team_id: str = Field(
        ...,
        description=(
            "The NBA Team ID (e.g. '1610612739' for Cleveland Cavaliers). "
            "Use other search tools or static data to find this ID."
        )
    )
    season: str = Field(
        default="2022-23",
        description=(
            "Season in 'YYYY-YY' format (e.g. '2022-23')."
        )
    )
    season_type: str = Field(
        default="Regular Season",
        description=(
            "One of 'Regular Season', 'Pre Season', 'Playoffs', or 'All Star'. "
            "Typically 'Regular Season'."
        )
    )

# 2) Define the Tool Class
class TeamGameLogsTool:
    """
    Fetches all game logs for a specific team in a certain season 
    using the `teamgamelogs.TeamGameLogs` endpoint from stats.nba.com.
    """
    def __init__(self):
        pass

    def run(self, team_id: str, season: str, season_type: str) -> List[Dict[str, Any]]:
        """
        Calls teamgamelogs.TeamGameLogs(...) and returns a simplified list 
        of dictionaries containing at least the 'GAME_ID' and other fields 
        like MATCHUP, GAME_DATE, W/L, etc.
        """
        try:
            # Use the TeamGameLogs endpoint
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable=season_type
            )
            # get_data_frames() returns a list of DataFrames. The main one is index=0
            df = logs.get_data_frames()[0]  # the primary DataFrame with all logs

            # Convert to dict. We'll select a few columns that matter for game identification
            # Feel free to keep or drop whichever columns you want.
            selected_columns = ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL"]
            partial_df = df[selected_columns]

            # Convert to list of dict
            results = partial_df.to_dict("records")
            return results
        except Exception as e:
            # Return a list with an error
            return [{"error": str(e)}]

# 3) Create the LangChain StructuredTool
team_game_logs_tool = StructuredTool(
    name="nba_team_game_logs",
    description=(
        "Fetch a list of all games (including game IDs, date, matchup, result) "
        "for a given Team ID in a specified season and season type. "
        "Useful to find all the game_ids a team played, from which you can pick a certain matchup."
    ),
    func=TeamGameLogsTool().run,
    args_schema=TeamGameLogsInput
)

# -------------------------------------------------------------------
# 11) team_game_logs_by_name_tool: Fetch a Team's Game Logs by Name
# -------------------------------------------------------------------

from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamgamelogs

# 1) Define Input Schema
class TeamGameLogsByNameInput(BaseModel):
    """
    User provides:
    - team_name: partial or full name for an NBA team (e.g. "Warriors", "Golden State Warriors")
    - season: e.g. "2022-23"
    - season_type: "Regular Season", "Playoffs", "Pre Season", or "All Star"
    """
    team_name: str = Field(
        ...,
        description="Partial or full NBA team name (e.g. 'Warriors', 'Cavaliers')."
    )
    season: str = Field(
        default="2022-23",
        description="Season in YYYY-YY format (e.g. '2022-23')."
    )
    season_type: str = Field(
        default="Regular Season",
        description="One of 'Regular Season', 'Playoffs', 'Pre Season', or 'All Star'."
    )

# 2) Define the Tool Class
class TeamGameLogsByNameTool:
    """
    Single tool that:
      1. Finds the best match for the given team name.
      2. Retrieves that team's ID.
      3. Calls 'teamgamelogs.TeamGameLogs' to fetch the logs (GAME_ID, MATCHUP, etc.).
    """
    def __init__(self):
        pass

    def run(self, team_name: str, season: str, season_type: str) -> List[Dict[str, Any]]:
        try:
            # A) Search teams by name
            found = teams.find_teams_by_full_name(team_name)

            if not found:
                return [{
                    "error": f"No NBA team found matching name '{team_name}'."
                }]
            elif len(found) > 1:
                # If you want to handle multiple matches differently, do so here.
                # Example: pick the first
                best_match = found[0]
            else:
                best_match = found[0]

            # B) Extract the team_id from best_match
            team_id = best_match["id"]  # e.g. 1610612744 for Golden State

            # C) Get the game logs from teamgamelogs
            logs = teamgamelogs.TeamGameLogs(
                team_id_nullable=str(team_id),
                season_nullable=season,
                season_type_nullable=season_type
            )

            df = logs.get_data_frames()[0]
            # We'll pick out some columns for clarity
            columns_we_want = ["TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL"]
            partial_df = df[columns_we_want]
            results = partial_df.to_dict("records")

            return results
        except Exception as e:
            return [{"error": str(e)}]

# 3) Create the LangChain StructuredTool
team_game_logs_by_name_tool = StructuredTool(
    name="nba_team_game_logs_by_name",
    description=(
        "Fetch a team's game logs (and thus game_ids) by providing the team name, "
        "without needing the numeric team_id directly. Returns a list of dictionaries "
        "with 'GAME_ID', 'GAME_DATE', 'MATCHUP', and 'WL'."
    ),
    func=TeamGameLogsByNameTool().run,
    args_schema=TeamGameLogsByNameInput
)



# ---------------------------------------------------- SOCCER ------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------


###############################################################################
# 1) get_league_id_by_name_tool: Retrieve the League ID by Name
###############################################################################

class GetLeagueIdByNameInput(BaseModel):
    """
    Input schema for retrieving the league ID based on the league name.
    """
    league_name: str = Field(
        ...,
        description="Name of the league (e.g. 'Premier League', 'La Liga')."
    )

class GetLeagueIdByNameTool:
    """
    1. Search for the league ID via /leagues?search=league_name.
    2. Return the league ID for the specified league name.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_league_id(self, league_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        try:
            # Step 1: Get league ID by searching for league name
            leagues_url = f"{self.base_url}/leagues"
            leagues_params = {"search": league_name}  # Search the league by name
            resp = requests.get(leagues_url, headers=headers, params=leagues_params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("response"):
                return {"error": f"No leagues found matching '{league_name}'."}
            
            # Grab the first league from the response (assuming there's only one match)
            league_id = data["response"][0]["league"]["id"]
            return {"league_id": league_id}

        except Exception as e:
            return {"error": str(e)}

# Define the tool to retrieve the league ID
get_league_id_by_name_tool = StructuredTool(
    name="get_league_id_by_name",
    description="Retrieve the league ID for a given league name (e.g. 'Premier League', 'La Liga').",
    func=GetLeagueIdByNameTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_league_id,
    args_schema=GetLeagueIdByNameInput
)


###############################################################################
# 1) GetStandingsTool: Retrieve League/Team Standings
###############################################################################

class GetStandingsToolInput(BaseModel):
    """
    Input schema for retrieving league/team standings.
    'season' is required by the API. 'league' or 'team' can be used.
    """
    league: Optional[int] = Field(
        default=None,
        description="League ID to retrieve standings for."
    )
    season: int = Field(
        ...,
        description="(REQUIRED) 4-digit season (e.g. 2021)."
    )
    team: Optional[int] = Field(
        default=None,
        description="Optionally retrieve standings for a specific team ID within that league/season."
    )

class GetStandingsTool:
    """
    Retrieves standings for a league or for a specific team in that league.
    Endpoint: GET /standings
    Docs: https://www.api-football.com/documentation-v3#operation/get-standings
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_standings(
        self,
        league: Optional[int],
        season: int,
        team: Optional[int]
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/standings"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        params = {"season": season}

        if league is not None:
            params["league"] = league
        if team is not None:
            params["team"] = team

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

get_standings_tool = StructuredTool(
    name="get_standings",
    description=(
        "Use this tool to retrieve the standings table of a league and season, "
        "optionally filtered by a team ID."
    ),
    func=GetStandingsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_standings,
    args_schema=GetStandingsToolInput
)


###############################################################################
# GetPlayerProfileTool: Fetch a Player's Profile
###############################################################################

class GetPlayerProfileInput(BaseModel):
    """
    Input schema for retrieving a player's profile by last name.
    """
    player_name: str = Field(
        ...,
        description="The last name of the player to look up. Must be >= 3 characters."
    )

class GetPlayerProfileTool:
    """
    Retrieves a player's profile and basic info by searching for their last name.
    Internally calls /players/profiles with a 'search' parameter.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_player_profile(self, player_name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/players/profiles"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        params = {
            "search": player_name,
            "page": 1  # We fetch only the first page for simplicity
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

get_player_profile_tool = StructuredTool(
    name="get_player_profile",
    description=(
        "Use this tool to retrieve a single player's profile info by their last name. "
        "Example usage: Provide 'Messi' or 'Ronaldo' to look up that player's details."
    ),
    func=GetPlayerProfileTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_player_profile,
    args_schema=GetPlayerProfileInput
)



# -------------------------------------------------------------------
#  GetTeamFixturesTool: Fetch a Team's Fixtures
# -------------------------------------------------------------------
class GetTeamFixturesInput(BaseModel):
    """
    Input for retrieving a team's fixtures by name.
    """
    team_name: str = Field(
        ...,
        description="The team's name to search for. Must be >= 3 characters for accurate search."
    )
    type: str = Field(
        default="upcoming",
        description="Either 'past' or 'upcoming' fixtures."
    )
    limit: int = Field(
        default=5,
        description="How many fixtures to retrieve: e.g. last=5 or next=5. Default=5."
    )

class GetTeamFixturesTool:
    """
    Given a team name, finds the team's ID, then fetches either the last N or next N fixtures.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_team_fixtures(self, team_name: str, type: str, limit: int) -> Dict[str, Any]:
        """
        1) Look up team ID from /teams?search={team_name}.
        2) Depending on 'type':
            - if 'past': use /fixtures?team=ID&last={limit}
            - if 'upcoming': use /fixtures?team=ID&next={limit}
        3) Return the resulting fixtures or an error if not found.
        """
        # Step 1: Find the Team ID
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        search_url = f"{self.base_url}/teams"
        search_params = {"search": team_name}

        try:
            search_resp = requests.get(search_url, headers=headers, params=search_params, timeout=15)
            search_resp.raise_for_status()
            teams_data = search_resp.json()

            if not teams_data.get("response"):
                return {"error": f"No teams found matching '{team_name}'."}

            # Just pick the first matching team for simplicity
            first_team = teams_data["response"][0]
            team_id = first_team["team"]["id"]

            # Step 2: Fetch fixtures
            fixtures_url = f"{self.base_url}/fixtures"
            fixtures_params = {"team": team_id}

            if type.lower() == "past":
                fixtures_params["last"] = limit
            else:
                # Default is 'upcoming'
                fixtures_params["next"] = limit

            fixtures_resp = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
            fixtures_resp.raise_for_status()
            return fixtures_resp.json()

        except Exception as e:
            return {"error": str(e)}

get_team_fixtures_tool = StructuredTool(
    name="get_team_fixtures",
    description=(
        "Given a team name, returns either the last N or the next N fixtures for that team. "
        "Useful for quickly seeing a team's recent or upcoming matches."
    ),
    func=GetTeamFixturesTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_team_fixtures,
    args_schema=GetTeamFixturesInput
)



# -------------------------------------------------------------------
# GetFixtureStatisticsTool: Fetch Detailed Fixture Stats
# -------------------------------------------------------------------
class GetFixtureStatisticsInput(BaseModel):
    """
    Input schema for retrieving a single fixture's detailed stats.
    """
    fixture_id: int = Field(
        ...,
        description="The numeric ID of the fixture/game. Example: 215662."
    )

class GetFixtureStatisticsTool:
    """
    Given a fixture (game) ID, retrieves stats like shots on goal, possession, corners, etc.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_fixture_stats(self, fixture_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/fixtures/statistics"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        params = {"fixture": fixture_id}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

get_fixture_statistics_tool = StructuredTool(
    name="get_fixture_statistics",
    description=(
        "Use this tool to retrieve box-score style statistics for a given fixture. "
        "You must already know the fixture ID, e.g. 215662."
    ),
    func=GetFixtureStatisticsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_fixture_stats,
    args_schema=GetFixtureStatisticsInput
)



# -------------------------------------------------------------------
# GetTeamFixturesByDateRangeTool
# -------------------------------------------------------------------
class GetTeamFixturesByDateRangeInput(BaseModel):
    team_name: str = Field(
        ...,
        description="Team name to search for (e.g. 'Arsenal', 'Barcelona')."
    )
    season: str = Field(
        default="2024",
        description="Season in YYYY format (e.g. '2024')."
    )
    from_date: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format (e.g. '2023-08-01')."
    )
    to_date: str = Field(
        ...,
        description="End date in YYYY-MM-DD format (e.g. '2023-08-31')."
    )

class GetTeamFixturesByDateRangeTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # RapidAPI base URL
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_team_fixtures_by_date_range(self, team_name: str, from_date: str, to_date: str, season: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        # Step 1: find team ID
        teams_url = f"{self.base_url}/teams"
        teams_params = {"search": team_name}
        resp = requests.get(teams_url, headers=headers, params=teams_params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # print(data)
        if not data.get("response"):
            return {"error": f"No team found matching '{team_name}'."}
        team_id = data["response"][0]["team"]["id"]

        # Step 2: fetch fixtures in date range
        fixtures_url = f"{self.base_url}/fixtures"
        fixtures_params = {
            "team": team_id,
            "from": from_date,
            "to": to_date,
            "season": season  # or some 4-digit year
        }
        resp_fixtures = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
        resp_fixtures.raise_for_status()
        return resp_fixtures.json()


get_team_fixtures_by_date_range_tool = StructuredTool(
    name="get_team_fixtures_by_date_range",
    description=(
        "Retrieve all fixtures for a given team within a date range. "
        "Input: team name, from_date (YYYY-MM-DD), to_date (YYYY-MM-DD)."
    ),
    func=GetTeamFixturesByDateRangeTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_team_fixtures_by_date_range,
    args_schema=GetTeamFixturesByDateRangeInput
)



# -------------------------------------------------------------------
# GetFixtureEventsTool
# -------------------------------------------------------------------
class GetFixtureEventsInput(BaseModel):
    fixture_id: int = Field(
        ...,
        description="Numeric ID of the fixture whose events you want (e.g. 215662)."
    )

class GetFixtureEventsTool:
    """
    Given a fixture ID, returns the events that occurred (goals, substitutions, cards, etc.).
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_fixture_events(self, fixture_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/fixtures/events"
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        params = {"fixture": fixture_id}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

get_fixture_events_tool = StructuredTool(
    name="get_fixture_events",
    description=(
        "Retrieve all in-game events for a given fixture ID (e.g. goals, cards, subs). "
        "You must know the fixture ID beforehand."
    ),
    func=GetFixtureEventsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_fixture_events,
    args_schema=GetFixtureEventsInput
)


# -------------------------------------------------------------------
# GetMultipleFixturesStatsTool
# -------------------------------------------------------------------
class GetMultipleFixturesStatsInput(BaseModel):
    fixture_ids: list[int] = Field(
        ...,
        description="A list of numeric fixture IDs to get stats for, e.g. [215662, 215663]."
    )

class GetMultipleFixturesStatsTool:
    """
    Given multiple fixture IDs, calls /fixtures/statistics for each ID one by one
    and aggregates the results in a list.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_multiple_fixtures_stats(self, fixture_ids: list[int]) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }
        combined_results = []

        for f_id in fixture_ids:
            try:
                url = f"{self.base_url}/fixtures/statistics"
                params = {"fixture": f_id}
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                combined_results.append({f_id: data})
            except Exception as e:
                combined_results.append({f_id: {"error": str(e)}})

        return {"fixtures_statistics": combined_results}

get_multiple_fixtures_stats_tool = StructuredTool(
    name="get_multiple_fixtures_stats",
    description=(
        "Retrieve stats (shots, possession, etc.) for multiple fixtures at once. "
        "Input a list of fixture IDs, e.g. [215662, 215663]."
    ),
    func=GetMultipleFixturesStatsTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_multiple_fixtures_stats,
    args_schema=GetMultipleFixturesStatsInput
)


# -------------------------------------------------------------------
# GetLeagueScheduleByDateTool
# -------------------------------------------------------------------
from langchain.tools.base import StructuredTool
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import requests

class GetLeagueScheduleByDateInput(BaseModel):
    league_name: str = Field(
        ...,
        description="Name of the league (e.g. 'Premier League', 'La Liga')."
    )
    date: str = Field(
        ...,
        description="Date in YYYY-MM-DD format (e.g. '2023-08-10')."
    )
    season: str = Field(
        default="2024",
        description="Season in YYYY format (e.g. '2024')."
    )

class GetLeagueScheduleByDateTool:
    """
    1. Search for the league ID via /leagues?search=league_name
    2. Use the found ID to call /fixtures?league={id}&date={YYYY-MM-DD}&season={season}
    3. Return JSON of the fixtures (the schedule for that day).
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_league_schedule(self, league_name: str, date: str, season: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        # Step 1: Get league ID by searching name
        try:
            leagues_url = f"{self.base_url}/leagues"
            leagues_params = {"search": league_name}  # You can adjust the season here
            resp = requests.get(leagues_url, headers=headers, params=leagues_params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("response"):
                return {"error": f"No leagues found matching '{league_name}'."}
            
            # We'll just grab the first result
            league_id = data["response"][0]["league"]["id"]
            # Step 2: Get fixtures for that league & date
            fixtures_url = f"{self.base_url}/fixtures"
            fixtures_params = {
                "league": league_id,
                "date": date,
                "season": season  # Adding the season parameter
            }
            resp_fixtures = requests.get(fixtures_url, headers=headers, params=fixtures_params, timeout=15)
            resp_fixtures.raise_for_status()
            return resp_fixtures.json()

        except Exception as e:
            return {"error": str(e)}

# Define the tool
get_league_schedule_by_date_tool = StructuredTool(
    name="get_league_schedule_by_date",
    description=(
        "Retrieve the schedule (fixtures) for a given league on a specified date. "
        "Input the league name (e.g. 'Premier League') and a date (YYYY-MM-DD)."
    ),
    func=GetLeagueScheduleByDateTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_league_schedule,
    args_schema=GetLeagueScheduleByDateInput
)

# -------------------------------------------------------------------
# GetLiveMatchForTeamTool: 
# -------------------------------------------------------------------
from langchain.tools.base import StructuredTool
from pydantic import BaseModel, Field
from typing import Any, Dict
import requests

class GetLiveMatchForTeamInput(BaseModel):
    """
    Minimal input: just the team's name.
    """
    team_name: str = Field(
        ...,
        description="The team's name. Example: 'Arsenal', 'Barcelona'. Must be >= 3 chars for accurate searching."
    )

class GetLiveMatchForTeamTool:
    """
    1) Resolve the team name to team ID (via /teams?search).
    2) Check /fixtures?team=TEAM_ID&live=all to find any in-progress match.
    3) Return the fixture data if live, else error message.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_live_match_for_team(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        # Step 1: find team ID
        try:
            teams_resp = requests.get(
                f"{self.base_url}/teams",
                headers=headers,
                params={"search": team_name},
                timeout=15
            )
            teams_resp.raise_for_status()
            teams_data = teams_resp.json()

            if not teams_data.get("response"):
                return {"error": f"No team found matching '{team_name}'."}

            team_id = teams_data["response"][0]["team"]["id"]

            # Step 2: look for live matches
            fixtures_resp = requests.get(
                f"{self.base_url}/fixtures",
                headers=headers,
                params={"team": team_id, "live": "all"},
                timeout=15
            )
            fixtures_resp.raise_for_status()
            fixtures_data = fixtures_resp.json()

            live_fixtures = fixtures_data.get("response", [])

            if not live_fixtures:
                return {"message": f"No live match found for '{team_name}' right now."}

            # Typically only 1, but if multiple, just return the first
            return {"live_fixture": live_fixtures[0]}

        except Exception as e:
            return {"error": str(e)}

get_live_match_for_team_tool = StructuredTool(
    name="get_live_match_for_team",
    description=(
        "Check if a given team is currently playing live. Input the team name. "
        "Returns the live match fixture info if found, else returns a message that no live match is found."
    ),
    func=GetLiveMatchForTeamTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_live_match_for_team,
    args_schema=GetLiveMatchForTeamInput
)

# -------------------------------------------------------------------
# GetLiveStatsForTeamTool
# -------------------------------------------------------------------
class GetLiveStatsForTeamInput(BaseModel):
    team_name: str = Field(
        ...,
        description="Team name to get live stats for. e.g. 'Arsenal', 'Barcelona'."
    )

class GetLiveStatsForTeamTool:
    """
    1. Find team ID by name.
    2. Find current live fixture for that team.
    3. If found, call /fixtures/statistics?fixture=FIXTURE_ID to get live stats.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_live_stats_for_team(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        try:
            # Step 1: get team ID
            teams_resp = requests.get(
                f"{self.base_url}/teams",
                headers=headers,
                params={"search": team_name},
                timeout=15
            )
            teams_resp.raise_for_status()
            teams_data = teams_resp.json()
            if not teams_data.get("response"):
                return {"error": f"No team found matching '{team_name}'."}
            team_id = teams_data["response"][0]["team"]["id"]

            # Step 2: check for live fixtures
            fixtures_resp = requests.get(
                f"{self.base_url}/fixtures",
                headers=headers,
                params={"team": team_id, "live": "all"},
                timeout=15
            )
            fixtures_resp.raise_for_status()
            fixtures_data = fixtures_resp.json()
            live_fixtures = fixtures_data.get("response", [])
            if not live_fixtures:
                return {"message": f"No live match for '{team_name}' right now."}

            fixture_id = live_fixtures[0]["fixture"]["id"]

            # Step 3: get stats for that fixture
            stats_resp = requests.get(
                f"{self.base_url}/fixtures/statistics",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=15
            )
            stats_resp.raise_for_status()
            stats_data = stats_resp.json()

            return {"fixture_id": fixture_id, "live_stats": stats_data}

        except Exception as e:
            return {"error": str(e)}

get_live_stats_for_team_tool = StructuredTool(
    name="get_live_stats_for_team",
    description=(
        "Retrieve live in-game stats (shots on goal, possession, etc.) for a team currently in a match. "
        "Input the team name. If no live match is found, returns a message."
    ),
    func=GetLiveStatsForTeamTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_live_stats_for_team,
    args_schema=GetLiveStatsForTeamInput
)

# -------------------------------------------------------------------
# GetLiveMatchTimelineTool
# -------------------------------------------------------------------
class GetLiveMatchTimelineInput(BaseModel):
    team_name: str = Field(
        ...,
        description="Team name to retrieve live timeline of the current match if playing. E.g. 'Arsenal'."
    )

class GetLiveMatchTimelineTool:
    """
    1. Find the team ID by name
    2. Check if there's a live fixture for that team
    3. If found, call /fixtures/events?fixture=... to get timeline events
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"

    def get_live_match_timeline(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",  # RapidAPI host
            "x-rapidapi-key": self.api_key                         # RapidAPI key
        }

        try:
            # Step 1: team ID
            teams_resp = requests.get(
                f"{self.base_url}/teams",
                headers=headers,
                params={"search": team_name},
                timeout=15
            )
            teams_resp.raise_for_status()
            teams_data = teams_resp.json()
            if not teams_data.get("response"):
                return {"error": f"No team found matching '{team_name}'."}
            team_id = teams_data["response"][0]["team"]["id"]

            # Step 2: check live fixtures
            fixtures_resp = requests.get(
                f"{self.base_url}/fixtures",
                headers=headers,
                params={"team": team_id, "live": "all"},
                timeout=15
            )
            fixtures_resp.raise_for_status()
            fixtures_data = fixtures_resp.json()
            live_fixtures = fixtures_data.get("response", [])
            if not live_fixtures:
                return {"message": f"No live match for '{team_name}' right now."}

            fixture_id = live_fixtures[0]["fixture"]["id"]

            # Step 3: get events timeline
            events_resp = requests.get(
                f"{self.base_url}/fixtures/events",
                headers=headers,
                params={"fixture": fixture_id},
                timeout=15
            )
            events_resp.raise_for_status()
            events_data = events_resp.json()

            return {"fixture_id": fixture_id, "timeline_events": events_data}

        except Exception as e:
            return {"error": str(e)}

get_live_match_timeline_tool = StructuredTool(
    name="get_live_match_timeline",
    description=(
        "Retrieve the real-time timeline of a currently live match for a given team. "
        "Input the team name. Returns events like goals, substitutions, and cards."
    ),
    func=GetLiveMatchTimelineTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_live_match_timeline,
    args_schema=GetLiveMatchTimelineInput
)

# -------------------------------------------------------------------
# LeagueInformationTool
# -------------------------------------------------------------------
class GetLeagueInfoInput(BaseModel):
    league_name: str = Field(..., description="Name of the league (e.g., 'Champions League')")

class GetLeagueInfoTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
    
    def get_league_info(self, league_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key
        }

        # Fetch league information
        league_url = f"{self.base_url}/leagues"
        params = {"search": league_name}
        resp = requests.get(league_url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data

# Define the tool
get_league_info_tool = StructuredTool(
    name="get_league_info",
    description="Retrieve information about a specific football league (teams, season, fixtures, etc.)",
    func=GetLeagueInfoTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_league_info,
    args_schema=GetLeagueInfoInput
)

# -------------------------------------------------------------------
# TeamInformationTool
# -------------------------------------------------------------------
class GetTeamInfoInput(BaseModel):
    team_name: str = Field(..., description="Name of the team (e.g., 'Manchester United')")

class GetTeamInfoTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api-football-v1.p.rapidapi.com/v3"
    
    def get_team_info(self, team_name: str) -> Dict[str, Any]:
        headers = {
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
            "x-rapidapi-key": self.api_key
        }

        # Fetch team information
        teams_url = f"{self.base_url}/teams"
        teams_params = {"search": team_name}
        resp = requests.get(teams_url, headers=headers, params=teams_params)
        resp.raise_for_status()
        data = resp.json()
        return data


# Define the tool
get_team_info_tool = StructuredTool(
    name="get_team_info",
    description="Retrieve basic information about a specific football team (players, history, etc.)",
    func=GetTeamInfoTool(api_key=os.getenv("RAPID_API_KEY_FOOTBALL")).get_team_info,
    args_schema=GetTeamInfoInput
)

# -------------------------------------------------------------------
# PlayerStatisticsTool
# -------------------------------------------------------------------


