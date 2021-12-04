# -*- coding: utf-8 -*-
"""
Oracle's Elixir Data Generator.

This script is intended to represent the main function to update data once per day.
Right now, it is storing data locally, but in the future it may write a NoSQL db
to AWS DyanmoDB where it can be leveraged by other querying services.

This script is intended to be kicked off by a Cron job on a daily basis at 7 AM.

Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""
# Housekeeping
import datetime as dt
import lolmodeling as lol
import configurations as conf
import oracleselixir as oe
import pandas as pd
import sys
from typing import Tuple


# Function Definitions
def enrich_dataset(player_data: pd.DataFrame,
                   team_data: pd.DataFrame,
                   directory: str = conf.workingdir) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all enrichments for team and player-based analytics and predictions.
    This includes DraftKings point totals, Team and Player-based elo, TrueSkill, and EGPM dominance.

    Parameters
    ----------
    player_data : pd.DataFrame
        DataFrame representing the output of oe.clean_data() split by player.
    team_data : pd.DataFrame
        DataFrame representing the output of oe.clean_data() split by team.
    directory : str
        String representing the file path of the working directory.

    Returns
    -------
    team_data: pd.DataFrame
        DataFrame containing team-based metrics and enrichments.
    player_data: pd.DataFrame
        DataFrame containing player-based metrics and enrichments.
    """
    initial_players = len(player_data)
    initial_teams = len(team_data)

    # Enrich DraftKings Points Data
    team_data = lol.dk_enrich(team_data, entity='team')
    player_data = lol.dk_enrich(player_data, entity='player')

    # EnrichElo Statistics
    player_data = lol.player_elo(player_data)

    team_data = lol.team_elo(team_data)
    team_data = lol.aggregate_player_elos(player_data, team_data)

    # Enrich Team TrueSkill
    player_data, team_data = lol.trueskill_model(player_data, team_data)

    # EGPM Model - TrueSkill Normalized Earned Gold
    team_data = lol.egpm_model(team_data, "team")
    player_data = lol.egpm_model(player_data, "player")

    # EWM Model - Side Win Rates
    team_data = lol.ewm_model(team_data, "team")
    player_data = lol.ewm_model(player_data, "player")

    # Data Validation
    final_players = len(player_data)
    print(f"Player Check: {final_players - initial_players}")

    final_teams = len(team_data)
    print(f"Team Check: {final_teams - initial_teams}")

    # Render CSV Files
    team_data.drop('index', axis=1, inplace=True)
    team_data.to_csv(f'{directory}\\ModelData\\team_data.csv', index=False)
    player_data.drop('index', axis=1, inplace=True)
    player_data.to_csv(f'{directory}\\ModelData\\player_data.csv', index=False)

    # Flatten Data Frame / Render
    team_data = team_data.sort_values(['teamid', 'date']).reset_index(drop=True)
    flattened_teams = team_data.groupby('teamid').nth(-1).reset_index(drop=True)
    flattened_teams = flattened_teams[["date", "teamname",
                                       "teamelo_after", "trueskill_sum_mu",
                                       "trueskill_sum_sigma", "egpm_dominance_ema",
                                       "blue_side_ema", "red_side_ema"]]
    flattened_teams = flattened_teams.rename(columns={'teamelo_after': 'team_elo'})
    flattened_teams.to_csv(f'{directory}\\ModelData\\flattened_teams.csv', index=False)

    player_data = player_data.sort_values(['playerid', 'date']).reset_index(drop=True)
    flattened_players = player_data.groupby('playerid').nth(-1).reset_index(drop=True)
    flattened_players = flattened_players[["date", "teamname", "position",
                                           "playername", "player_elo_after", "trueskill_mu",
                                           "trueskill_sigma", "egpm_dominance_ema",
                                           "blue_side_ema", "red_side_ema"]]
    flattened_players = flattened_players.rename(columns={'player_elo_after': 'player_elo'})
    flattened_players.to_csv(f'{directory}\\ModelData\\flattened_players.csv', index=False)

    return team_data, player_data


def main():
    # Define time frame for analytics
    current_year = dt.date.today().year
    years = [str(current_year), str(current_year - 1)]

    # Download Data
    data = oe.download_data(directory=conf.workingdir, years=years)

    # Remove Buggy Matches (both red/blue team listed as same team, invalid for elo/TrueSkill)
    invalid_games = ['NA1/3754345055', 'NA1/3754344502',
                     'ESPORTSTMNT02/1890835', 'NA1/3669212337',
                     'NA1/3669211958', 'ESPORTSTMNT02/1890848']
    data = data[~data.gameid.isin(invalid_games)].copy()

    # Clean/Format Data
    teams = oe.clean_data(data, split_on='team',
                          team_replacements=conf.team_replacements,
                          player_replacements=conf.player_replacements)

    players = oe.clean_data(data, split_on='player',
                            team_replacements=conf.team_replacements,
                            player_replacements=conf.player_replacements)

    # Enrich Data
    teams, players = enrich_dataset(player_data=players, team_data=teams)

    return teams, players


if __name__ in ('__main__', '__builtin__', 'builtins'):
    main()
    print("Data Generated.")