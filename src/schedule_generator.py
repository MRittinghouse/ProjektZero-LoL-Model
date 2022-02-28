import datetime as dt
from dotenv import load_dotenv
import json
import numpy as np
from os import getenv
import pandas as pd
from pathlib import Path
import requests
from src.team import Team
from src.match_predictor import predict_match
from typing import Optional

# Variable Definitions
load_dotenv()
panda_key = getenv("PANDASCORE_KEY")


# Helper/Utility Functions:
def __team_namer(team_name: str) -> str:
    """
    Rename teams to ensure consistency between Oracles Elixir data and the schedule API.

    Parameters
    ----------
    team_name : str
        Name of team from the data set.
    Returns
    -------
    str
        The correct team name.
    """

    # Misnomers Format: (API Name) : (Flattened Teams Name)
    misnomers = {
        "Schalke 04": "FC Schalke 04 Esports",
        "Edward Gaming": "EDward Gaming",
        "kt Rolster": "KT Rolster",
        "TT": "ThunderTalk Gaming",
        "Thunder Talk Gaming": "ThunderTalk Gaming",
        "NONGSHIM REDFORCE": "Nongshim RedForce",
        "NongShim REDFORCE": "Nongshim RedForce",
        "Nongshim Red Force": "Nongshim RedForce",
        "EXCEL": "Excel Esports",
        "Dignitas QNTMPAY": "Dignitas",
        "Immortals Progressive": "Immortals",
        "Team SoloMid": "TSM",
        "Team SoloMid Academy": "TSM Academy",
        "Team SoloMid Amateur": "TSM Amateur",
        "BDS": "Team BDS",
        "BDS Academy": "Team BDS Academy",
        "INTZ e-Sports": "INTZ",
        "EDward Gaming Youth Team": "EDG Youth Team",
        "Istanbul Wildcats": "İstanbul Wildcats",
        "KaBuM! eSports": "KaBuM! e-Sports",
        "MAX E-Sports Club": "MAX",
        "Hive Athens": "Hive Athens EC",
        "Komil&Friends": "Komil&amp;Friends",
        "GG&Esports": "GGEsports",
        "UCAM Esports Club": "UCAM Tokiers",
        "We Love Gaming": "WLGaming Esports",
        "⁠Entropiq": "Entropiq",
    }

    if team_name in misnomers.keys():
        return misnomers[team_name]
    else:
        return team_name


def __schedule_predictor(blue_name: str, red_name: str):
    """
    Parameters
    ----------
    blue_name: str
        String containing the name of blue team, matching OE data.
    red_name: str
        String containing the name of the red team, matching OE data.

    Returns
    -------
    probability: int
        Likelihood that the blue team will win.
    """
    try:
        blue = Team(name=blue_name, side="Blue")
        red = Team(name=red_name, side="Red")
        match = predict_match(blue, red)
        match = match[["blue_win_chance", "deviation"]]
        return match["blue_win_chance"].iloc[0], match["deviation"].iloc[0]
    except Exception as e:
        print(f"Error: {blue_name}, {red_name}, {e}")
        return np.nan, np.nan


# Primary Functions
def pandascore_schedule(api_key: str, leagues: Optional[str], days: int) -> pd.DataFrame:
    """
    Pings the Pandascore API endpoint to grab a dictionary of upcoming matches.
    The Pandascore API is public and has a free tier, so this is something anyone can use.
    You are responsible for getting your own key and adding it to your own .env file.

    Parameters
    ----------
    leagues : str or list of strings
        An optional string or list of strings containing Leagues of interest. This API supports many leagues.
    api_key : str
        The private key used to credential your access to the Pandascore API.
    days : int
        Number of days forward to search for matches (today + X days).
    Returns
    -------
    upcoming : pd.DataFrame
        A Pandas DataFrame with the upcoming matches.
    """
    schedule = []
    pages = range(1, 6)  # TODO: Make this pull until max date observed instead of a set number of pages.
    headers = {"Accept": "application/json"}

    for i in pages:
        url = f"https://api.pandascore.co/lol/matches/upcoming?sort=&page={i}&per_page=100&token={api_key}"

        # Make API Request
        try:
            res = requests.get(url, headers=headers)
        except ConnectionError as e:
            res = {'status_code': e}

        # Get Schedule or Error Response
        if res.status_code == 200:
            pandascore_response = json.loads(res.text)
        else:
            raise ConnectionError(f"Error: {res.status_code}")

        for match in pandascore_response:
            try:
                match_data = {"league": match["league"]["name"],
                              "Blue": match["opponents"][0]["opponent"]["name"],
                              "Red": match["opponents"][1]["opponent"]["name"],
                              "Start (UTC)": match["scheduled_at"],
                              "Best Of": match["number_of_games"]}
                schedule.append(match_data)
            except IndexError:
                print(match)  # Match does not have opponents yet (e.g. playoffs before teams locked in)

    # Filter Schedule of Interest
    if leagues:
        leagues = [leagues] if isinstance(leagues, str) else leagues
        schedule = list(filter(lambda game: game["league"] in leagues, schedule))

    # Filter Games By Date Range
    days = int(days)
    current_date = dt.datetime.now()
    future_date = current_date + dt.timedelta(days=days)
    tformat = "%Y-%m-%dT%H:%M:%SZ"

    upcoming = (list(filter(
        lambda game: future_date >= dt.datetime.strptime(game["Start (UTC)"], tformat) >= current_date, schedule)))

    # If this block of code fails, there's a problem with the team names
    for i in upcoming:
        if i["Blue"] == "TBD" or i["Red"] == "TBD":
            upcoming.remove(i)
        i.update({"Blue": __team_namer(i["Blue"]), "Red": __team_namer(i["Red"])})

    upcoming = pd.DataFrame(upcoming)

    return upcoming


def main():
    filepath = Path.cwd().parent
    schedule = pandascore_schedule(leagues=None, api_key=panda_key, days=5)

    if isinstance(schedule, pd.DataFrame):
        predictions = schedule.apply(lambda row: __schedule_predictor(row["Blue"], row["Red"]), axis=1)
        matches = pd.concat([schedule, predictions], axis=1)
        matches[["Blue Win%", "Deviation"]] = pd.DataFrame(matches[0].tolist(), index=matches.index).round(4)
        matches = matches.drop(0, axis=1).reset_index(drop=True)
        matches.to_csv(filepath.joinpath('data', 'processed', 'schedule.csv'), index=False)
    else:
        print("No upcoming Pandascore match data available.")


if __name__ in ('__main__', '__builtin__', 'builtins'):
    start = dt.datetime.now()
    main()
    end = dt.datetime.now()
    elapsed = end - start
    print(f"Schedule generated in {elapsed}.")
