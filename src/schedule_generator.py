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
import sys
from typing import Optional

# Variable Definitions
load_dotenv()
key = getenv("OE_SCHEDULE_KEY")
url = getenv("OE_SCHEDULE_URL")


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

    # Misnomers Format: (API Name) : (OE Name)
    misnomers = {
        "Schalke 04": "FC Schalke 04 Esports",
        "Edward Gaming": "EDward Gaming",
        "kt Rolster": "KT Rolster",
        "TT": "ThunderTalk Gaming",
        "Thunder Talk Gaming": "ThunderTalk Gaming",
        "NONGSHIM REDFORCE": "Nongshim RedForce",
        "NongShim REDFORCE": "Nongshim RedForce",
        "EXCEL": "Excel Esports",
        "Dignitas QNTMPAY": "Dignitas",
        "Immortals Progressive": "Immortals"
    }

    if team_name in misnomers.keys():
        return misnomers[team_name]
    else:
        return team_name


def schedule_predictor(blue_name: str, red_name: str):
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
def upcoming_schedule(leagues: Optional[str],
                      api_url: str, api_key: str, days: int = 7) -> dict:
    """
    Pings an API endpoint to grab a dictionary of upcoming matches.
    Please note, this API has not yet been released, so if you do not have
    a valid key, do not try to use this function.

    Parameters
    ----------
    leagues : str or list of strings
        A string or list of strings containing Leagues of interest
        Valid Leagues are LCS, LEC, LCK, LPL.
        The API does not support all leagues, stick to the big four.
    api_url : str
        The URL of the API to pull data from.
    api_key : str
        The private key used to credential your access to the API.
    days : int
        Number of days forward to search for matches (today + X days).
    Returns
    -------
    upcoming : pd.DataFrame
        A Pandas DataFrame with the upcoming matches.
    """
    # Get API Response
    headers = {"x-api-key": f"{api_key}"}
    try:
        res = requests.get(api_url, headers=headers)
    except ConnectionError as e:
        res = {'status_code': e}

    # Get Schedule or Error Response
    if res.status_code == 200:
        schedule = json.loads(res.text)
    else:
        raise ConnectionError(f"Error: {res.status_code}")

    # Filter Schedule of Interest
    if leagues:
        leagues = [leagues] if isinstance(leagues, str) else leagues
        schedule = list(filter(lambda game: game["league"] in leagues, schedule))

    # Filter Games By Date Range
    days = int(days)
    current_date = dt.datetime.now()
    future_date = current_date + dt.timedelta(days=days)
    tformat = "%Y-%m-%dT%H:%M:%S.%fZ"

    upcoming = (list(filter(
        lambda game: future_date >= dt.datetime.strptime(game["startTime"], tformat) >= current_date,
        schedule)))

    # If this block of code fails, there's a problem with the team names
    for i in upcoming:
        if i["team1Code"] == "TBD" or i["team2Code"] == "TBD":
            upcoming.remove(i)
        i["team1Name"] = __team_namer(i["team1Name"])
        i["team2Name"] = __team_namer(i["team2Name"])

    upcoming = pd.DataFrame(upcoming)

    return upcoming


def main():
    filepath = Path.cwd().parent

    matches = upcoming_schedule(leagues=None,
                                api_url=url, api_key=key,
                                days=5)

    if isinstance(matches, pd.DataFrame):
        matches = matches[["league", "team1Name", "team2Name", "startTime", "seriesType"]]
        matches = matches.rename(columns={'team1Name': 'Blue',
                                          'team2Name': 'Red',
                                          'startTime': 'Start (UTC Time)',
                                          'seriesType': 'Type'})
        predictions = matches.apply(lambda row: schedule_predictor(row["Blue"], row["Red"]),
                                    axis=1)
        matches = pd.concat([matches, predictions], axis=1)
        matches[["Blue Win%", "Deviation"]] = pd.DataFrame(matches[0].tolist(), index=matches.index).round(4)
        matches = matches.drop(0, axis=1).reset_index(drop=True)
        matches.to_csv(filepath.joinpath('data', 'processed', 'schedule.csv'), index=False)
        return matches
    else:
        print("Nothing to see here. Move along.")


if __name__ in ('__main__', '__builtin__', 'builtins'):
    main()
    print("Schedule generated.")
