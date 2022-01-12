import datetime as dt
from dotenv import load_dotenv
import json
from os import getenv
import pandas as pd
from pathlib import Path
import requests
from typing import Optional

# Variable Definitions
load_dotenv()
key = getenv("OE_SCHEDULE_KEY")
url = getenv("OE_SCHEDULE_URL")

"""
Misnomers Format: (API Name) : (OE Name)
This will need to be regularly monitored and updated.
If you get key errors, update this dictionary.
"""
misnomers = {
    "Schalke 04": "FC Schalke 04 Esports",
    "Edward Gaming": "EDward Gaming",
    "kt Rolster": "KT Rolster",
    "TT": "ThunderTalk Gaming",
    "NONGSHIM REDFORCE": "Nongshim RedForce",
    "EXCEL": "Excel Esports",
    "Dignitas": "Dignitas QNTMPAY",
}


# Helper/Utility Functions:
def __team_namer(team_name: str, misnomer_lookup: dict) -> str:
    """
    Rename teams to ensure consistency between Oracles Elixir data and the schedule API.

    Parameters
    ----------
    team_name : str
        Name of team from the data set.
    misnomer_lookup : dict
        Dictionary object containing misspellings/other names from the data.
    Returns
    -------
    str
        The correct team name.
    """
    if team_name in misnomer_lookup.keys():
        return misnomer_lookup[team_name]
    else:
        return team_name


# Primary Functions
def upcoming_schedule(leagues: Optional[str], misnomer_lookup: dict,
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
    misnomer_lookup: dict
        A dictionary containing lookup values to correct inconsistencies between API and OE.
    api_url : str
        The URL of the API to pull data from.
    api_key : str
        The private key used to credential your access to the API.
    days : int
        Number of days forward to search for matches (today + X days).
    Returns
    -------
    upcoming : dict
        A dictionary with the format of {ID: [blue, red]} for upcoming matches.
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

    upcoming = pd.DataFrame(upcoming)

    return upcoming


def main():
    filepath = Path.cwd().parent

    matches = upcoming_schedule(leagues=None,
                                misnomer_lookup=misnomers,
                                api_url=url, api_key=key,
                                days=6)

    if isinstance(matches, pd.DataFrame):
        matches = matches[["league", "team1Name", "team2Name", "startTime", "seriesType"]]
        matches = matches.rename(columns={'team1Name': 'Blue',
                                          'team2Name': 'Red',
                                          'startTime': 'Start (UTC Time)',
                                          'seriesType': 'Type'})
        matches.to_csv(filepath.joinpath('data', 'processed', 'schedule.csv'), index=False)
        return matches
    else:
        print("Nothing to see here. Move along.")


if __name__ in ('__main__', '__builtin__', 'builtins'):
    main()
