import requests
import datetime

class BettingAPI:
    """
    Base class for common API functionality.
    """
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def make_request(self, endpoint, params=None):
        """Helper function to make API requests."""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return None


class OddsAPI(BettingAPI):
    """
    Class for interacting with the Odds API.
    """
    def __init__(self, api_key):
        super().__init__(api_key, "https://api.the-odds-api.com/v4/sports/")

    def get_active_events(self, sport_name):
        """Retrieve active events for a specific sport."""
        endpoint = ""
        params = {"apiKey": self.api_key}
        return self.make_request(endpoint, params=params)

    def get_event_odds(self, sport_key, region="uk", odds_format="decimal"):
        """Retrieve odds for a specific event."""
        endpoint = f"{sport_key}/odds"
        params = {
            "regions": region,
            "oddsFormat": odds_format,
            "apiKey": self.api_key,
        }
        return self.make_request(endpoint, params=params)

    def get_event_scores(self, sport_key):
        """Retrieve scores for a specific event."""
        endpoint = f"{sport_key}/scores"
        params = {"apiKey": self.api_key}
        return self.make_request(endpoint, params=params)


class BetsAPI(BettingAPI):
    """
    Class for interacting with the Bets API.
    """
    def __init__(self, api_key):
        super().__init__(api_key, "https://api.b365api.com/v1/")

    def get_active_events(self, sport_id):
        """Retrieve active events for a specific sport."""
        endpoint = f"bet365/inplay_filter"
        params = {"sport_id": sport_id, "token": self.api_key}
        return self.make_request(endpoint, params=params)

    def get_event_details(self, event_id):
        """Retrieve details of a specific event."""
        endpoint = f"bet365/event"
        params = {"FI": event_id, "token": self.api_key}
        return self.make_request(endpoint, params=params)

    def get_upcoming_events(self, sport_id):
        """Retrieve upcoming events for a specific sport."""
        endpoint = f"bet365/upcoming"
        params = {"sport_id": sport_id, "token": self.api_key}
        return self.make_request(endpoint, params=params)


class BettingManager:
    """
    Manager class to handle API interactions and data aggregation.
    """
    def __init__(self, odds_api_key, bets_api_key):
        self.odds_api = OddsAPI(odds_api_key)
        self.bets_api = BetsAPI(bets_api_key)

    def get_combined_active_events(self, sport_name, sport_id):
        """Combine active events from OddsAPI and BetsAPI."""
        odds_events = self.odds_api.get_active_events(sport_name)
        bets_events = self.bets_api.get_active_events(sport_id)

        return {
            "OddsAPI Events": odds_events,
            "BetsAPI Events": bets_events,
        }

    def get_live_odds(self, sport_key, event_id):
        """Fetch live odds for a given sport and event."""
        odds = self.odds_api.get_event_odds(sport_key)
        bets = self.bets_api.get_event_details(event_id)

        return {
            "OddsAPI Odds": odds,
            "BetsAPI Odds": bets,
        }


# Example usage:
odds_api_key = "your_odds_api_key"
bets_api_key = "your_bets_api_key"
manager = BettingManager(odds_api_key, bets_api_key)

# Fetch combined active events for Cricket (sport_name: Cricket, sport_id: 3)
active_events = manager.get_combined_active_events("Cricket", 3)
print(active_events)

# Fetch live odds for a specific match
live_odds = manager.get_live_odds("cricket_psl", "153088197")
print(live_odds)
