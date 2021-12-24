"""
Oracle's Elixir

This script is designed to connect to Tim Sevenhuysen's Oracle's Elixir site to pull down and format data.
It is built to empower esports enthusiasts, data scientists, or anyone to leverage pro game data
for use in their own scripts and analytics.

Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""
# Housekeeping
import datetime as dt
import io
import numpy as np
import os
import pandas as pd
from pathlib import Path
import requests
from typing import Optional, Union


# Utility/Helper Function Definitions
def _get_opponent(column: pd.Series, entity: str) -> list:
    """
    Generate value for the opposing team or player.
    This can be used for utilities such as returning the opposing player or team's name.
    It can also be used to return opposing metrics, such as opponent's earned gold per minute, etc.
    Be sure that the input value is sorted to have consistent order of matches/positions.

    Parameters
    ----------
    column : Pandas Series
        Pandas DataFrame column representing entity data (see entity)
    entity : str
        'player' or 'team', entity to calculate opponent of.

    Returns
    -------
    opponent : list
        The opponent of the entities in the column provided; can be inserted as a column back into the dataframe.
    """
    opponent = []
    flag = 0

    # The gap represents how many rows separate a value from its opponent
    # Teams are 1 (Team A, Team B)
    # Players are 5 (ADC to opposing ADC is a 5 row gap)
    if entity == "player":
        gap = 5
    elif entity == "team":
        gap = 1
    else:
        raise ValueError("Entity must be either player or team.")

    for i, obj in enumerate(column):
        # If "Blue Side" - fetch opposing team/player below
        if flag < gap:
            opponent.append(column[i + gap])
            flag += 1
        # If "Red Side" - fetch opposing team/player above
        elif gap <= flag < (gap * 2):
            opponent.append(column[i - gap])
            flag += 1
        else:
            raise ValueError(f"Index {i} - Out Of Bounds")

        # After both sides are enumerated, reset the flag
        if flag >= (gap * 2):
            flag = 0
    return opponent


# Primary Functions
def download_data(years: Optional[Union[list, str, int]] = [dt.date.today().year],
                  delete: Optional[bool] = True) -> pd.DataFrame:
    r"""
    Download game data from Oracle's Elixir.
    This interface will help set up the directory for you, remove old data files, and pull the latest data.
    The data will be automatically saved for you as a .csv file, and the function returns a Pandas dataframe
    that is ready for use in additional analytics.
    If up-to-date local data is already present, this function will simply import that to reduce download volumes.

    Parameters
    ----------
    delete : boolean
        If True, will delete files in directory upon download of new data.
    years : Union[list, str, int]
        A string or list of strings containing years (e.g. ["2019", "2020"])
        If nothing is specified, returns the current year only by default.

    Returns
    -------
    A Pandas dataframe containing the most recent Oracle's Elixir data
    for the years provided by the year parameter.
    A .csv file in the directory, with the most recent data, if downloaded.
    """
    # Defining Time Variables
    directory = Path.cwd().parent.joinpath('data', 'raw')

    current_date = dt.date.today()
    today = current_date.strftime("%Y%m%d")
    yesterday = current_date - dt.timedelta(days=1)
    yesterday = yesterday.strftime("%Y%m%d")

    # Other Variables
    url = ("https://oracleselixir-downloadable-match-data."
           "s3-us-west-2.amazonaws.com/")
    oe_data = pd.DataFrame()

    # Conditional Handling For Years
    if isinstance(years, (str, int)):
        years = [years]

    # Dynamic Data Import
    for year in years:
        file = f"{year}_LoL_esports_match_data_from_OraclesElixir_"
        current_files = [x for x in os.listdir(directory) if x.startswith(file)]

        if f"{file}{today}.csv" not in current_files:
            # If today's data not in Dir, optionally delete old versions
            if delete:
                for f in current_files:
                    print(f"DELETED: {f}")
                    os.remove(directory.joinpath(f))
            try:
                # Try To Grab File For Current Date
                filepath = f"{url}{file}{today}.csv"

                r = requests.get(filepath, allow_redirects=True)
                data = r.content.decode("utf8")
                data = pd.read_csv(io.StringIO(data), low_memory=False)
                assert len(data) > 9
                data.to_csv(directory.joinpath(f"{file}{today}.csv"), index=False)
                print(f"DOWNLOADED: {file}{today}.csv")
            except Exception as e:
                # Grab Yesterday's Data If Today's Does Not Exist
                filepath = f"{url}{file}{yesterday}.csv"

                r = requests.get(filepath, allow_redirects=True)
                data = r.content.decode("utf8")
                data = pd.read_csv(io.StringIO(data), low_memory=False)
                assert len(data) > 9
                data.to_csv(directory.joinpath(f"{file}{yesterday}.csv"), index=False)
                print(e)
                print(f"DOWNLOADED: {file}{yesterday}.csv")
        else:
            # Grab Local Data If It Exists
            data = pd.read_csv(directory.joinpath(f"{file}{today}.csv"), low_memory=False)
            print(f"USING LOCAL {year} DATA")

        # Concatenate Data To Master Data Frame
        oe_data = pd.concat([oe_data, data], axis=0)

    return oe_data


def clean_data(oe_data: pd.DataFrame,
               split_on: Optional[str],
               team_replacements: Optional[dict] = None,
               player_replacements: Optional[dict] = None) -> pd.DataFrame:
    """
    Format and clean data from Oracle's Elixir.
    This function is optional, and provided as a convenience to help make the data more consistent and user-friendly.

    The date column will be formatted appropriately as a datetime object.
    Only games with datacompletness = complete will be kept.
    Any games with 'unknown team' or 'unknown player' will be dropped.
    Any games with null game ids will be dropped.
    Opponent metrics will be enriched into the dataframe.
    This function also subsets the dataset down to relevant columns for the entity you split on (team, player).
    Please note that this means not all columns from the initial data set are in the "cleaned" output.

    Parameters
    ----------
    oe_data : DataFrame
        A Pandas data frame containing Oracle's Elixir data.
    split_on : 'team', 'player' or None
        Subset data for Team data or Player data. None for all data.
    team_replacements: Optional[dict]
        Replacement values to normalize team names in the data if a team name changes over time.
        The format is intended to be {'oldname1': 'newname1', 'oldname2': 'newname2'}
    player_replacements: Optional[dict]
        Replacement values to normalize player names in the data if a player's name changes over time.
        The format is intended to be {'oldname1': 'newname1', 'oldname2': 'newname2'}

    Returns
    -------
    A Pandas dataframe of formatted, subset Oracle's Elixir data matching
    the parameters provided above.
    """
    # Preliminary Data Type Formatting
    oe_data = oe_data.astype({"date": "datetime64",
                              "gameid": "str",
                              "playerid": "str",
                              "teamid": "str"})

    # Keep Only "Complete" Games
    oe_data = oe_data[oe_data["datacompleteness"] != "partial"].copy()
    oe_data = oe_data.drop(columns=["datacompleteness"], axis=1)

    # Format IDs & Remove Any Games With Null GameIDs
    oe_data["gameid"] = oe_data["gameid"].str.strip()
    oe_data["playerid"] = oe_data["playerid"].str.strip()
    oe_data["teamid"] = oe_data["teamid"].str.strip()

    replace_values = {"": np.nan, "nan": np.nan, "null": np.nan}
    oe_data = oe_data.replace({"gameid": replace_values,
                               "playerid": replace_values,
                               "teamid": replace_values,
                               "position": replace_values})

    # Remove Any Records With Null Data In Critical Keys
    oe_data = oe_data[oe_data.gameid.notna()]
    oe_data = oe_data[oe_data.position.notna()]

    # Drop Games With "Unknown Player/Team" Lookup Failures
    drop_games = oe_data[(oe_data["playername"] == "unknown player") | (oe_data["teamname"] == "unknown team")].copy()
    drop_games = drop_games["gameid"].unique()
    oe_data = (oe_data[~oe_data.gameid.isin(drop_games)].copy().reset_index(drop=True))

    # Normalize Player/Team Names
    if team_replacements:
        oe_data['teamname'] = oe_data['teamname'].replace(team_replacements)
    if player_replacements:
        oe_data['playername'] = oe_data['playername'].replace(player_replacements)

    # Split On Player/Team (and defining some related variables)
    if split_on == "team":
        oe_data = oe_data[oe_data["position"] == "team"]
        cap = 2
        oe_data = oe_data[["date", "gameid", "side", "league", "teamname", "teamid",
                           "result", "kills", "deaths", "assists",
                           "earned gpm", "gamelength", "ckpm", "team kpm",
                           "firstblood", "dragons", "barons", "towers",
                           "goldat15", "xpat15", "csat15",
                           "golddiffat15", "xpdiffat15", "csdiffat15"]]
    elif split_on == "player":
        oe_data = oe_data[oe_data["position"] != "team"].copy()
        cap = 10
        oe_data = oe_data[["date", "gameid", "side", "position", "league",
                           "playername", "playerid", "teamname", "teamid", "result", "kills",
                           "deaths", "assists", "total cs", "earned gpm", "earnedgoldshare",
                           "gamelength", "ckpm", "team kpm",
                           "goldat15", "xpat15", "csat15", "killsat15",
                           "assistsat15", "deathsat15", "opp_killsat15",
                           "opp_assistsat15", "opp_deathsat15", "golddiffat15",
                           "xpdiffat15", "csdiffat15"]]
    else:
        raise ValueError("Must split on either player or team.")

    # Remove Games That Don't Have Data For All Players OR Have Data For Too Many Players
    counts = oe_data["gameid"].value_counts().to_frame()
    counts = counts[(counts["gameid"] < cap) | (counts["gameid"] > cap)]
    if len(counts) > 0:
        drop_games = counts.index.to_list()
        oe_data = oe_data[~oe_data.gameid.isin(drop_games)].copy().reset_index(drop=True)

    # Sort Values To Ensure Consistent Data
    if split_on == "player":
        split_name = "playername"
        split_id = "playerid"
        oe_data = oe_data.sort_values(["league", "date", "gameid", "side", "position"])

    elif split_on == "team":
        split_name = "teamname"
        split_id = "teamid"
        oe_data = oe_data.sort_values(["league", "date", "gameid", "side"])

    else:
        raise ValueError("Must split on either player or team.")

    # SAFETY - Fill Null Values For Player/Team ID With Player Name:
    # The invocation of this at the player level is in the if split_on == "player" section below.
    oe_data["teamid"] = oe_data["teamid"].fillna(oe_data["teamname"])

    # Enrich Opponent-Based Metrics
    if split_on == "player":
        oe_data["playerid"] = oe_data["playerid"].fillna(oe_data["playername"])
        oe_data["opponentteam"] = _get_opponent(oe_data.teamname.to_list(), split_on)
        oe_data["opponentteamid"] = _get_opponent(oe_data.teamid.to_list(), split_on)

    oe_data["opponentname"] = _get_opponent(oe_data[split_name].to_list(), split_on)
    oe_data["opponentid"] = _get_opponent(oe_data[split_id].to_list(), split_on)
    oe_data["opponent_egpm"] = _get_opponent(oe_data["earned gpm"].to_list(), split_on)

    # Return Output
    return oe_data
