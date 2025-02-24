import requests

class OddsAPI:
    """Base class to interact with The Odds API"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"

    def _make_request(self, endpoint, params):
        params['api_key'] = self.api_key
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error during request: {e}")
            return None

    def get_in_season_sports(self):
        """Retrieve a list of in-season sports."""
        return self._make_request("sports", {})

    def get_events(self, sport_key):
        """Retrieve live and upcoming events for a sport."""
        endpoint = f"sports/{sport_key}/events"
        return self._make_request(endpoint, {})

    def get_odds(self, sport_key, regions, markets, odds_format="decimal", date_format="iso"):
        """Retrieve odds for events."""
        endpoint = f"sports/{sport_key}/odds"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        return self._make_request(endpoint, params)

    def get_historical_odds(self, sport_key, event_id, date, regions, markets, odds_format="decimal", date_format="iso"):
        """Retrieve historical odds for a specific event."""
        endpoint = f"historical/sports/{sport_key}/events/{event_id}/odds"
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
            "date": date,
        }
        return self._make_request(endpoint, params)


class OddsDataProcessor:
    """Class to process odds data."""

    @staticmethod
    def american_to_decimal(american_odds):
        if american_odds < 0:
            return 1 - (100 / american_odds)
        return 1 + (american_odds / 100)

    @staticmethod
    def decimal_to_american(decimal_odds):
        if decimal_odds == 1:
            return 0
        if decimal_odds < 2:
            return int(round(100 / (1 - decimal_odds), 0))
        return int(round(100 * (decimal_odds - 1), 0))

    @staticmethod
    def find_most_balanced(outcomes, american_format=True):
        side_1, side_2 = outcomes
        side_1_by_point = {o['point']: o for o in side_1}
        side_2_by_point = {o['point']: o for o in side_2}

        differences = {}
        for point in side_1_by_point:
            if point not in side_2_by_point:
                continue
            s1_price = side_1_by_point[point]['price']
            s2_price = side_2_by_point[point]['price']

            if american_format:
                s1_price = OddsDataProcessor.american_to_decimal(s1_price)
                s2_price = OddsDataProcessor.american_to_decimal(s2_price)

            differences[point] = abs(s1_price - s2_price)

        most_balanced_point = min(differences, key=differences.get)
        return side_1_by_point[most_balanced_point], side_2_by_point[most_balanced_point]


# Example usage
if __name__ == "__main__":
    api_key = "6d881c9632eddaa1df6e218e1e24c5ef"
    odds_api = OddsAPI(api_key)

    # Retrieve in-season sports
    sports = odds_api.get_in_season_sports()
    print("In-season sports:", sports)

    # Retrieve events for a specific sport
    sport_key = "basketball_nba"
    events = odds_api.get_events(sport_key)
    print(f"Events for {sport_key}:", events)

    # Retrieve odds for a specific event
    odds = odds_api.get_odds(sport_key, "us", "h2h,totals", "american")
    print(f"Odds for {sport_key}:", odds)
