# -*- coding: utf-8 -*-
"""
League of Legends Modeling Tools.

This script is intended to work off of Tim Sevenhuysen's Oracle's Elixir data.
Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.

This code is intended to be imported into other Python analytics projects
It provides a number of tools for statistics and analytics, with the focus
of generating a probability model to predict win rates between teams.

Please note that this is purely for academic interests, and this script
comes with no guarantee or expectation of performance, and any use of it
for betting/wagers is done entirely at the risk of the end user.
Nothing in this code or its outputs constitutes financial advice.

This code is intended to be imported into your analytics projects.
"""
# Housekeeping
import csv
import itertools
import math
import numpy as np
import src.oracles_elixir as oe
import pandas as pd
from scipy.stats import norm
import trueskill
from typing import Optional


# Function Library
def std(x: int) -> int:
    """
    Calculate the standard deviation of X using Numpy."""
    return np.std(x)


def q10(x: pd.Series) -> pd.Series:
    """Calculate the 10th percentile of a Numpy series."""
    return np.percentile(x, q=10)


def q90(x: pd.Series) -> pd.Series:
    """Calculate the 90th percentile of a Numpy series."""
    return np.percentile(x, q=90)


def elo_calculator(df: pd.DataFrame, entity: str,
                   start_elo: int = 1200, k: int = 28,
                   league_modifiers: Optional[dict] = None) -> pd.DataFrame:
    """
    Calculate elo from Pandas dataframe.

    Attributes
    ----------
        df: pd.DataFrame
            Must contain winning team ['team'] and losing opponent ['opponent'] in order by game date
        entity: str
             The entity to calculate elo upon (team or player)
        start_elo: int
            Identifies the starting Elo (default=1200)
        k: int
            The k-factor for the elo calculation (default=28)
        league_modifiers: dict
            Dictionary containing League names as the keys and weights from 0 to 1 as values.
    """
    # Variable Definitions
    elo_results = {}
    elo_expected = []
    result_expected = []
    elo_winner_after = []
    elo_winner_before = []
    elo_loser_after = []
    elo_loser_before = []

    # Function Definitions
    def get_elo(team: str) -> dict:
        return elo_results.get(team, start_elo)

    def expected_result(elo_a: int, elo_b: int) -> int:
        expect_a = 1.0 / (1 + 10 ** ((elo_b - elo_a) / 400))
        return expect_a

    def update_elo(winner: str, loser: str):
        winner_elo = get_elo(winner)
        elo_winner_before.append(winner_elo)
        loser_elo = get_elo(loser)
        elo_loser_before.append(loser_elo)
        result_expected.append(1 if winner_elo > loser_elo else 0)
        expected_win = expected_result(winner_elo, loser_elo)
        change_in_elo = k * (1 - expected_win)
        winner_elo += change_in_elo
        loser_elo -= change_in_elo
        elo_results[winner] = winner_elo
        elo_results[loser] = loser_elo
        elo_winner_after.append(winner_elo)
        elo_loser_after.append(loser_elo)
        elo_expected.append(expected_win)

    if entity == 'player':
        df.apply(lambda row: update_elo(row['playerid'], row['opponentid']), axis=1)
        elo_dataframe = pd.DataFrame({'league': df['league'],
                                      'gameid': df['gameid'],
                                      'date': df['date'],
                                      'win_team_ref': df['teamid'],
                                      'position': df['position'],
                                      'winner': df['playerid'],
                                      'winning_elo_before': elo_winner_before,
                                      'winning_elo_after': elo_winner_after,
                                      'win_perc': elo_expected,
                                      'expected_result': result_expected,
                                      'lose_team_ref': df['opponentteamid'],
                                      'loser': df['opponentid'],
                                      'losing_elo_before': elo_loser_before,
                                      'losing_elo_after': elo_loser_after
                                      })
    elif entity == 'team':
        df.apply(lambda row: update_elo(row['teamid'], row['opponentid']), axis=1)
        elo_dataframe = pd.DataFrame({'league': df['league'],
                                      'gameid': df['gameid'],
                                      'date': df['date'],
                                      'win_team_ref': df['teamname'],
                                      'winner': df['teamid'],
                                      'winning_elo_before': elo_winner_before,
                                      'winning_elo_after': elo_winner_after,
                                      'win_perc': elo_expected,
                                      'expected_result': result_expected,
                                      'lose_team_ref': df['opponentname'],
                                      'loser': df['opponentid'],
                                      'losing_elo_before': elo_loser_before,
                                      'losing_elo_after': elo_loser_after
                                      })
    else:
        raise ValueError('Entity must be either PLAYER or TEAM.')

    return elo_dataframe


def team_elo(df: pd.DataFrame, start: int = 1200, k: int = 28) -> pd.DataFrame:
    """
    Calculate team elo across a dataframe of Oracle's Elixir data.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing OE team data.
    start: Integer
        Integer value representing the starting / initial elo value.
    k: Integer
        Integer representing how much elo is gained/lost per match.
        Testing suggests that 28 is an appropriate score for League games.
        Please research how elo is calculated before you modify.

    Returns
    -------
    A Pandas dataframe containing the latest elo scores for the teams in the
        league specified within the leagues parameter.
    """
    df = df.sort_values(['date', 'league', 'gameid', 'result']).reset_index()
    win_df = df[df['result'] == 1]
    elo_data = elo_calculator(win_df, entity='team', start_elo=start, k=k)

    # Combine Data Frames To Generate Output
    winner_data = elo_data[['winner', 'win_team_ref', 'league', 'gameid', 'date',
                            'winning_elo_before', 'win_perc', 'winning_elo_after']].copy()
    winner_data['team_elo_diff'] = elo_data['winning_elo_before'] - elo_data['losing_elo_before']
    winner_data = winner_data.rename(columns={'winner': 'teamid',
                                              'win_team_ref': 'teamname',
                                              'winning_elo_before': 'team_elo_before',
                                              'win_perc': 'team_elo_win_perc',
                                              'winning_elo_after': 'team_elo_after'})

    loser_data = elo_data[['loser', 'lose_team_ref', 'league', 'gameid', 'date',
                           'losing_elo_before', 'losing_elo_after']].copy()
    loser_data['win_perc'] = 1 - elo_data['win_perc']
    loser_data['team_elo_diff'] = elo_data['losing_elo_before'] - elo_data['winning_elo_before']
    loser_data = loser_data.rename(columns={'loser': 'teamid',
                                            'lose_team_ref': 'teamname',
                                            'losing_elo_before': 'team_elo_before',
                                            'win_perc': 'team_elo_win_perc',
                                            'losing_elo_after': 'team_elo_after'})

    # Merge Things Back Together
    elo_data = pd.concat([winner_data, loser_data], ignore_index=True)
    df = (elo_data.merge(df, how='left',
                         left_on=['league', 'gameid', 'date', 'teamname', 'teamid'],
                         right_on=['league', 'gameid', 'date', 'teamname', 'teamid'])
          .reset_index(drop=True))
    df.sort_values(by=['date', 'league', 'gameid', 'result'], ascending=True, inplace=True)

    return df


def player_elo(df: pd.DataFrame, start: int = 1200, k: int = 28) -> pd.DataFrame:
    """
    Calculate elo for a player dataframe.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing OE player data.
    start: Integer
        Integer value representing the starting / initial elo value.
    k: Integer
        Integer representing how much elo is gained/lost per match.
        Testing suggests that 28 is an appropriate score for League games.
        Please research how elo is calculated before you modify.

    Returns
    -------
    A Pandas dataframe containing the elo scores per match by player.
    This dataframe is an expansion of the player data input.
    """
    df = df.sort_values(['date', 'league', 'gameid', 'teamid', 'position']).reset_index()
    win_df = df[df['result'] == 1]
    elo_data = elo_calculator(win_df, entity='player', start_elo=start, k=k)

    # Combine Data Frames To Generate Output
    winner_data = elo_data[['league', 'gameid', 'position', 'date',
                            'win_team_ref', 'winner', 'winning_elo_before', 'win_perc', 'winning_elo_after']].copy()
    winner_data['player_elo_diff'] = elo_data['winning_elo_before'] - elo_data['losing_elo_before']
    winner_data = winner_data.rename(columns={'win_team_ref': 'teamid',
                                              'winner': 'playerid',
                                              'winning_elo_before': 'player_elo_before',
                                              'win_perc': 'player_elo_win_perc',
                                              'winning_elo_after': 'player_elo_after'})
    loser_data = elo_data[['league', 'gameid', 'position', 'date',
                           'lose_team_ref', 'loser', 'losing_elo_before', 'losing_elo_after']].copy()
    loser_data['win_perc'] = 1 - elo_data['win_perc']
    loser_data['player_elo_diff'] = elo_data['losing_elo_before'] - elo_data['winning_elo_before']
    loser_data = loser_data.rename(columns={'lose_team_ref': 'teamid',
                                            'loser': 'playerid',
                                            'losing_elo_before': 'player_elo_before',
                                            'win_perc': 'player_elo_win_perc',
                                            'losing_elo_after': 'player_elo_after'})

    # Merge Things Back Together
    elo_data = pd.concat([winner_data, loser_data], ignore_index=True)

    df = (elo_data.merge(df, how='left',
                         left_on=['gameid', 'league', 'position', 'date', 'teamid', 'playerid'],
                         right_on=['gameid', 'league', 'position', 'date', 'teamid', 'playerid'])
          .reset_index(drop=True))
    df.sort_values(by=['date', 'league', 'gameid', 'result', 'position'], ascending=True, inplace=True)

    return df


def aggregate_player_elos(player_data: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual player elos to a mean elo per team per match.
    The output data is team-based and can be joined back to the team dataframe.

    Parameters
    ----------
    player_data : DataFrame
        DataFrame containing the output of the player_based_elo function.
    team_data: DataFrame
        DataFrame containing Oracle's Elixir data, split by team/match.

    Returns
    -------
    team_data : DataFrame
        DataFrame containing player-based elo for each team for each match.
    """
    # Aggregate Individual Players Into Teams
    player_data = player_data[['league', 'gameid', 'date', 'teamid',
                               'player_elo_before', 'player_elo_after', 'player_elo_diff', 'player_elo_win_perc']]
    team_elo_playerbased = (player_data.groupby(['league', 'gameid', 'teamid', 'date'])
                            .agg(['mean'])
                            .reset_index())
    team_elo_playerbased.columns = team_elo_playerbased.columns.droplevel(1)

    team_data = (team_data.merge(team_elo_playerbased, how='left',
                                 on=['gameid', 'league', 'date', 'teamid'])
                 .reset_index(drop=True))
    team_data.sort_values(by=['date', 'league', 'gameid', 'teamid'], ascending=True, inplace=True)

    return team_data


def trueskill_model(player_data: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
    r"""
    Calculate team ranking using Microsoft's TrueSkill 1 algorithm.
    Reference: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/

    Parameters
    ----------
    player_data : DataFrame
        Pandas DataFrame containing Oracle's Elixir player data.
    team_data : DataFrame
        Pandas DataFrame containing Oracle's Elixir team data.

    Returns
    -------
    A Pandas dataframe containing the latest TrueSkill scores for the TEAMS
        in the league specified within the leagues parameter.
    This will be an expanded version of the team_data input.
    """
    # Data Setup
    player_data = player_data.sort_values(['date', 'gameid', 'side', 'position']).reset_index()

    input_data = player_data[['gameid', 'date', 'league', 'playername', 'playerid', 'side',
                              'teamname', 'teamid', 'position', 'result', 'earned gpm',
                              'ckpm', 'team kpm']].copy()

    earned_gold = (input_data.groupby(['league', 'gameid', 'date', 'teamid'])
                   ['earned gpm'].sum()
                   .reset_index())
    earned_gold = earned_gold.rename(columns={'earned gpm': 'team_egpm'})

    input_data = pd.merge(input_data, earned_gold, how='left', on=['league', 'gameid', 'date', 'teamid'])
    input_data = input_data.rename(columns={'team kpm': 'team_kpm'})
    input_data[['team_egpm', 'team_kpm', 'ckpm']] = input_data[['team_egpm', 'team_kpm', 'ckpm']].fillna(0)

    # Initialize TrueSkill Player Ratings Dictionary
    ts = trueskill.TrueSkill(draw_probability=0.0)
    player_ratings_dict = dict()
    for i in input_data['playerid'].unique():
        player_ratings_dict[i] = ts.create_rating()

    def setup_match(df: pd.DataFrame) -> np.array:
        # Prepare DataFrame
        df = df.sort_values(['date', 'gameid', 'side', 'position']).reset_index()
        df = df[['playerid', 'playername', 'date', 'result', 'teamname',
                 'league', 'ckpm', 'gameid', 'team_egpm', 'team_kpm', 'teamid']].copy().values

        # Define Initial Variables
        matches_count = int(len(df) / 10)
        pointer = 0
        output_array = []

        for m in range(matches_count):
            # Define Index Position
            if pointer == 0:
                ind = m
            else:
                ind = (m * 10)

            # If you ever have to modify these, the index values correspond to the "df" from line 53.
            match_array = [df[ind, 7],  # gameid
                           df[ind, 2],  # date
                           df[ind, 5],  # league
                           df[ind, 6],  # ckpm
                           df[ind, 8],  # blue earned gpm
                           df[ind, 9],  # blue kpm
                           df[ind, 4],  # blue team name
                           df[ind, 10],  # blue team id
                           df[ind + 4, 0],  # blue top
                           df[ind + 1, 0],  # blue jng
                           df[ind + 2, 0],  # blue mid
                           df[ind, 0],  # blue bot
                           df[ind + 3, 0],  # blue sup
                           df[ind, 3],  # blue result
                           df[ind + 5, 4],  # red team name
                           df[ind + 5, 10],  # red team id
                           df[ind + 5, 8],  # red earned gpm
                           df[ind + 5, 9],  # red kpm
                           df[ind + 9, 0],  # red top
                           df[ind + 6, 0],  # red jng
                           df[ind + 7, 0],  # red mid
                           df[ind + 5, 0],  # red bot
                           df[ind + 8, 0]]  # red sup
            output_array.append(match_array)
            pointer += 1

        return output_array

    colnames = ['gameid', 'date', 'league', 'ckpm', 'blue_earned_gpm',
                'blue_kpm', 'blue_team', 'blue_team_id', 'blue_top_name', 'blue_jng_name',
                'blue_mid_name', 'blue_bot_name', 'blue_sup_name',
                'blue_team_result', 'red_team', 'red_team_id', 'red_earned_gpm', 'red_kpm',
                'red_top_name', 'red_jng_name', 'red_mid_name',
                'red_bot_name', 'red_sup_name']

    lcs_rating = pd.DataFrame(setup_match(input_data), columns=colnames)

    analyzed_gameids = {}

    def win_probability(team1: str, team2: str, trueskill_global_env):
        """
        Compute the TrueSkill probability of a team to win based on mu and sigma values.
        """
        delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
        sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
        size = len(team1) + len(team2)
        denominator = math.sqrt(size * (trueskill_global_env.beta ** 2) + sum_sigma)

        return trueskill_global_env.cdf(delta_mu / denominator)

    def update_trueskill(rating_dict, gameid_dict, gameid, blue_team_result,
                         blue_top_name, blue_jng_name, blue_mid_name,
                         blue_bot_name, blue_sup_name, red_top_name,
                         red_jng_name, red_mid_name, red_bot_name,
                         red_sup_name):
        """
        Compute individual changes to a player's mu and sigma values as a result of a given match.
        """
        rating_groups = [(rating_dict[blue_top_name],
                          rating_dict[blue_jng_name],
                          rating_dict[blue_mid_name],
                          rating_dict[blue_bot_name],
                          rating_dict[blue_sup_name]),
                         (rating_dict[red_top_name],
                          rating_dict[red_jng_name],
                          rating_dict[red_mid_name],
                          rating_dict[red_bot_name],
                          rating_dict[red_sup_name])]
        blue_mu = rating_groups[0][0].mu
        blue_sigma = rating_groups[0][0].sigma
        red_mu = rating_groups[1][0].mu
        red_sigma = rating_groups[1][0].sigma
        blue_team_win_prob = win_probability(rating_groups[0],
                                             rating_groups[1], ts)

        # Get Mu by position
        blue_top_mu = rating_dict[blue_top_name].mu
        blue_jng_mu = rating_dict[blue_jng_name].mu
        blue_mid_mu = rating_dict[blue_mid_name].mu
        blue_bot_mu = rating_dict[blue_bot_name].mu
        blue_sup_mu = rating_dict[blue_sup_name].mu
        red_top_mu = rating_dict[red_top_name].mu
        red_jng_mu = rating_dict[red_jng_name].mu
        red_mid_mu = rating_dict[red_mid_name].mu
        red_bot_mu = rating_dict[red_bot_name].mu
        red_sup_mu = rating_dict[red_sup_name].mu

        # Get Sigma by position
        blue_top_sigma = rating_dict[blue_top_name].sigma
        blue_jng_sigma = rating_dict[blue_jng_name].sigma
        blue_mid_sigma = rating_dict[blue_mid_name].sigma
        blue_bot_sigma = rating_dict[blue_bot_name].sigma
        blue_sup_sigma = rating_dict[blue_sup_name].sigma
        red_top_sigma = rating_dict[red_top_name].sigma
        red_jng_sigma = rating_dict[red_jng_name].sigma
        red_mid_sigma = rating_dict[red_mid_name].sigma
        red_bot_sigma = rating_dict[red_bot_name].sigma
        red_sup_sigma = rating_dict[red_sup_name].sigma

        # Update ratings
        if blue_team_result == 1:
            # For ranks, 0 represents the winner
            rated_rating_groups = ts.rate(rating_groups, ranks=[0, 1])
        else:
            rated_rating_groups = ts.rate(rating_groups, ranks=[1, 0])

        # Return values for new columns
        ts_update = pd.Series([blue_team_win_prob, blue_mu, blue_sigma, red_mu,
                               red_sigma, blue_top_mu, blue_top_sigma,
                               blue_jng_mu, blue_jng_sigma, blue_mid_mu,
                               blue_mid_sigma, blue_bot_mu, blue_bot_sigma,
                               blue_sup_mu, blue_sup_sigma, red_top_mu,
                               red_top_sigma, red_jng_mu, red_jng_sigma,
                               red_mid_mu, red_mid_sigma, red_bot_mu,
                               red_bot_sigma, red_sup_mu, red_sup_sigma])

        # Update the rating dictionary
        rating_dict[blue_top_name] = rated_rating_groups[0][0]
        rating_dict[blue_jng_name] = rated_rating_groups[0][1]
        rating_dict[blue_mid_name] = rated_rating_groups[0][2]
        rating_dict[blue_bot_name] = rated_rating_groups[0][3]
        rating_dict[blue_sup_name] = rated_rating_groups[0][4]
        rating_dict[red_top_name] = rated_rating_groups[1][0]
        rating_dict[red_jng_name] = rated_rating_groups[1][1]
        rating_dict[red_mid_name] = rated_rating_groups[1][2]
        rating_dict[red_bot_name] = rated_rating_groups[1][3]
        rating_dict[red_sup_name] = rated_rating_groups[1][4]

        # Conditional handling to prevent gameIDs from duplicative updating TS ratings
        if gameid in gameid_dict:
            return gameid_dict[gameid]
        else:
            gameid_dict[gameid] = ts_update
            return ts_update

    lcs_rating[['blue_team_win_prob', 'blue_mu', 'blue_sigma', 'red_mu',
                'red_sigma', 'blue_top_mu', 'blue_top_sigma', 'blue_jng_mu',
                'blue_jng_sigma', 'blue_mid_mu', 'blue_mid_sigma',
                'blue_bot_mu', 'blue_bot_sigma', 'blue_sup_mu',
                'blue_sup_sigma', 'red_top_mu', 'red_top_sigma',
                'red_jng_mu', 'red_jng_sigma', 'red_mid_mu', 'red_mid_sigma',
                'red_bot_mu', 'red_bot_sigma', 'red_sup_mu',
                'red_sup_sigma']] = lcs_rating.apply(
        lambda row: update_trueskill(
            player_ratings_dict, analyzed_gameids, row['gameid'],
            row['blue_team_result'], row['blue_top_name'],
            row['blue_jng_name'], row['blue_mid_name'],
            row['blue_bot_name'], row['blue_sup_name'],
            row['red_top_name'], row['red_jng_name'],
            row['red_mid_name'], row['red_bot_name'],
            row['red_sup_name']), axis=1)
    lcs_rating['blue_expected_result'] = np.where(lcs_rating['blue_team_win_prob'] >= 0.5, 1, 0)

    # Merge New Information Into Team Data
    blue_mu = ['blue_top_mu', 'blue_jng_mu', 'blue_mid_mu',
               'blue_bot_mu', 'blue_sup_mu']
    blue_sigma = ['blue_top_sigma', 'blue_jng_sigma',
                  'blue_mid_sigma', 'blue_bot_sigma', 'blue_sup_sigma']
    red_mu = ['red_top_mu', 'red_jng_mu', 'red_mid_mu',
              'red_bot_mu', 'red_sup_mu']
    red_sigma = ['red_top_sigma', 'red_jng_sigma', 'red_mid_sigma',
                 'red_bot_sigma', 'red_sup_sigma']

    blue = lcs_rating[['gameid', 'date', 'blue_team', 'blue_team_id', 'blue_team_win_prob']].copy()
    blue['blue_sum_mu'] = lcs_rating[blue_mu].sum(axis=1)
    blue['blue_sigma_squared'] = lcs_rating.apply(lambda row: sum([row[x] ** 2 for x in blue_sigma]), axis=1)
    blue['opponent_sum_mu'] = lcs_rating[red_mu].sum(axis=1)
    blue['opponent_sigma_squared'] = lcs_rating.apply(lambda row: sum([row[x] ** 2 for x in red_sigma]), axis=1)
    blue['trueskill_diff'] = blue['blue_team_win_prob'] - 0.50
    blue = blue.rename(columns={'blue_team': 'teamname',
                                'blue_team_id': 'teamid',
                                'blue_sum_mu': 'trueskill_sum_mu',
                                'blue_sigma_squared': 'trueskill_sigma_squared',
                                'blue_team_win_prob': 'trueskill_win_perc'})

    red = lcs_rating[['gameid', 'date', 'red_team', 'red_team_id']].copy()
    red['red_team_win_prob'] = 1 - lcs_rating['blue_team_win_prob']
    red['red_sum_mu'] = lcs_rating[red_mu].sum(axis=1)
    red['red_sigma_squared'] = lcs_rating.apply(lambda row: sum([row[x] ** 2 for x in red_sigma]), axis=1)
    red['opponent_sum_mu'] = lcs_rating[blue_mu].sum(axis=1)
    red['opponent_sigma_squared'] = lcs_rating.apply(lambda row: sum([row[x] ** 2 for x in blue_sigma]), axis=1)
    red['trueskill_diff'] = red['red_team_win_prob'] - 0.50
    red = red.rename(columns={'red_team': 'teamname',
                              'red_team_id': 'teamid',
                              'red_sum_mu': 'trueskill_sum_mu',
                              'red_sigma_squared': 'trueskill_sigma_squared',
                              'red_team_win_prob': 'trueskill_win_perc'})

    # Merge Things Back Together
    team_trueskill = pd.concat([blue, red], ignore_index=True)
    team_trueskill = team_trueskill.astype({"gameid": "str"})

    team_data = team_data.astype({"gameid": "str"})
    team_data = (pd.merge(left=team_data, right=team_trueskill, how='left',
                          left_on=['gameid', 'date', 'teamname', 'teamid'],
                          right_on=['gameid', 'date', 'teamname', 'teamid'])
                 .reset_index(drop=True))
    team_data.sort_values(by=['date', 'gameid', 'side'], ascending=True, inplace=True)

    # Merge New Information To Player Data
    # Blue
    blue_top = (lcs_rating[['gameid', 'date', 'blue_team_id', 'blue_top_name',
                            'blue_top_mu', 'blue_top_sigma']].copy()
                .rename(columns={'blue_team_id': 'teamid', 'blue_top_name': 'playerid',
                                 'blue_top_mu': 'trueskill_mu', 'blue_top_sigma': 'trueskill_sigma'}))
    blue_jng = (lcs_rating[['gameid', 'date', 'blue_team_id', 'blue_jng_name',
                            'blue_jng_mu', 'blue_jng_sigma']].copy()
                .rename(columns={'blue_team_id': 'teamid', 'blue_jng_name': 'playerid',
                                 'blue_jng_mu': 'trueskill_mu', 'blue_jng_sigma': 'trueskill_sigma'}))
    blue_mid = (lcs_rating[['gameid', 'date', 'blue_team_id', 'blue_mid_name',
                            'blue_mid_mu', 'blue_mid_sigma']].copy()
                .rename(columns={'blue_team_id': 'teamid', 'blue_mid_name': 'playerid',
                                 'blue_mid_mu': 'trueskill_mu', 'blue_mid_sigma': 'trueskill_sigma'}))
    blue_bot = (lcs_rating[['gameid', 'date', 'blue_team_id', 'blue_bot_name',
                            'blue_bot_mu', 'blue_bot_sigma']].copy()
                .rename(columns={'blue_team_id': 'teamid', 'blue_bot_name': 'playerid',
                                 'blue_bot_mu': 'trueskill_mu', 'blue_bot_sigma': 'trueskill_sigma'}))
    blue_sup = (lcs_rating[['gameid', 'date', 'blue_team_id', 'blue_sup_name',
                            'blue_sup_mu', 'blue_sup_sigma']].copy()
                .rename(columns={'blue_team_id': 'teamid', 'blue_sup_name': 'playerid',
                                 'blue_sup_mu': 'trueskill_mu', 'blue_sup_sigma': 'trueskill_sigma'}))

    # Red
    red_top = (lcs_rating[['gameid', 'date', 'red_team_id', 'red_top_name',
                           'red_top_mu', 'red_top_sigma']].copy()
               .rename(columns={'red_team_id': 'teamid', 'red_top_name': 'playerid',
                                'red_top_mu': 'trueskill_mu', 'red_top_sigma': 'trueskill_sigma'}))
    red_jng = (lcs_rating[['gameid', 'date', 'red_team_id', 'red_jng_name',
                           'red_jng_mu', 'red_jng_sigma']].copy()
               .rename(columns={'red_team_id': 'teamid', 'red_jng_name': 'playerid',
                                'red_jng_mu': 'trueskill_mu', 'red_jng_sigma': 'trueskill_sigma'}))
    red_mid = (lcs_rating[['gameid', 'date', 'red_team_id', 'red_mid_name',
                           'red_mid_mu', 'red_mid_sigma']].copy()
               .rename(columns={'red_team_id': 'teamid', 'red_mid_name': 'playerid',
                                'red_mid_mu': 'trueskill_mu', 'red_mid_sigma': 'trueskill_sigma'}))
    red_bot = (lcs_rating[['gameid', 'date', 'red_team_id', 'red_bot_name',
                           'red_bot_mu', 'red_bot_sigma']].copy()
               .rename(columns={'red_team_id': 'teamid', 'red_bot_name': 'playerid',
                                'red_bot_mu': 'trueskill_mu', 'red_bot_sigma': 'trueskill_sigma'}))
    red_sup = (lcs_rating[['gameid', 'date', 'red_team_id', 'red_sup_name',
                           'red_sup_mu', 'red_sup_sigma']].copy()
               .rename(columns={'red_team_id': 'teamid', 'red_sup_name': 'playerid',
                                'red_sup_mu': 'trueskill_mu', 'red_sup_sigma': 'trueskill_sigma'}))

    # Concat
    player_trueskill = pd.concat([blue_top, blue_jng, blue_mid, blue_bot, blue_sup,
                                  red_top, red_jng, red_mid, red_bot, red_sup], axis=0)
    player_trueskill.sort_values(by=['gameid', 'date', 'teamid'], ascending=True, inplace=True)
    player_data = (pd.merge(left=player_data, right=player_trueskill, how='left',
                            left_on=['gameid', 'date', 'teamid', 'playerid'],
                            right_on=['gameid', 'date', 'teamid', 'playerid'])
                   .reset_index(drop=True))

    player_data["opponent_mu"] = oe.get_opponent(player_data["trueskill_mu"].to_list(), "player")
    player_data["opponent_sigma"] = oe.get_opponent(player_data["trueskill_sigma"].to_list(), "player")
    player_data.sort_values(by=['date', 'league', 'gameid', 'teamname', 'position'], ascending=True, inplace=True)

    return player_data, team_data, player_ratings_dict


def ewm_model(df: pd.DataFrame, entity: str) -> pd.DataFrame:
    r"""
    Generate an Exponentially Weighted Mean (EWM) model for side win rates.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing Team Data as from oe_data import split on team.
    entity: str
        Entity to compute ewm modeling on ("Team" or "Player").

    Returns
    -------
    output : DataFrame
        A DataFrame containing the EWM model output for each team.
    """
    # Split Data
    if entity == "team":
        identity = "teamid"
    elif entity == "player":
        identity = "playerid"

    else:
        raise ValueError('Entity must be either PLAYER or TEAM.')
    df['result'] = df['result'].astype(float)

    # Compute EMA Columns
    red_side = df[df['side'] == 'Red'].copy()
    red_side['red_side_ema_before'] = (red_side.groupby(identity)['result']
                                       .transform(lambda x: x.ewm(halflife=5, ignore_na=True).mean().shift().bfill()))
    red_side['red_side_ema_after'] = (red_side.groupby(identity)['result']
                                      .transform(lambda x: x.ewm(halflife=5, ignore_na=True).mean()))

    blue_side = df[df['side'] == 'Blue'].copy()
    blue_side['blue_side_ema_before'] = (blue_side.groupby(identity)['result']
                                         .transform(lambda x: x.ewm(halflife=5, ignore_na=True).mean().shift().bfill()))
    blue_side['blue_side_ema_after'] = (blue_side.groupby(identity)['result']
                                        .transform(lambda x: x.ewm(halflife=5, ignore_na=True)
                                                   .mean()))

    # Merge Data
    merged = pd.concat([blue_side, red_side], ignore_index=True)
    merged = merged.sort_values([identity, 'date']).reset_index(drop=True)

    win_rates = (merged.groupby([identity], as_index=False)
                 .apply(lambda group: group.ffill())
                 .reset_index(drop=True))
    columns = ['blue_side_ema_before', 'blue_side_ema_after', 'red_side_ema_before', 'red_side_ema_after']
    win_rates[columns] = win_rates[columns].fillna(1)

    #  Compute Opponent Columns
    win_rates["opp_red_side_ema_before"] = oe.get_opponent(win_rates["red_side_ema_before"].to_list(), "team")
    win_rates["opp_blue_side_ema_before"] = oe.get_opponent(win_rates["blue_side_ema_before"].to_list(), "team")

    # Predict Win Probability By Side Win Rate
    win_rates["side_ema_before"] = (np.where(win_rates['side'] == 'Blue',
                                             win_rates["blue_side_ema_before"],
                                             win_rates["red_side_ema_before"]))
    win_rates["opp_side_ema_before"] = (np.where(win_rates['side'] == 'Blue',
                                                 win_rates["opp_red_side_ema_before"],
                                                 win_rates["opp_blue_side_ema_before"]))

    win_rates["side_ema_win_perc"] = (win_rates["side_ema_before"] /
                                      (win_rates["side_ema_before"] + win_rates["opp_side_ema_before"]))
    win_rates["side_ema_win_perc"] = win_rates["side_ema_win_perc"].fillna(0.5)

    # Final Sort
    win_rates.sort_values(by=['date', 'league', 'gameid', 'result'], ascending=True, inplace=True)

    return win_rates


def egpm_model(data: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Calculate a model based on Earned GPM with TrueSkill factored in.

    Parameters
    ----------
    data : DataFrame
        Oracle's Elixir data as provided by the output of the Team TrueSkill function.
    entity : str
        Entity to compute on, either "Team" or "Player"

    Returns
    -------
    DataFrame with Earned Gold Per Minute vs Opponent Team Strength model.
    """
    # Data Setup
    if entity == 'team':
        identity = 'teamid'
        elo = 'team_elo_before'
        opp_elo = data['team_elo_before'] - data['team_elo_diff']
    elif entity == 'player':
        identity = 'playerid'
        elo = 'player_elo_before'
        opp_elo = data['player_elo_before'] - data['player_elo_diff']
    else:
        raise ValueError('Entity must be either PLAYER or TEAM.')

    data['egpm_dominance_ratio'] = (data['earned gpm'] /
                                    (opp_elo /
                                     data[elo]))
    data['opp_egpm_dominance_ratio'] = (data['opponent_egpm'] /
                                        (data[elo] /
                                         opp_elo))
    data['egpm_dominance_ema_before'] = (data.groupby([identity])
                                         ['egpm_dominance_ratio']
                                         .transform(lambda x: x.ewm(halflife=9, ignore_na=True)
                                                    .mean().shift().bfill()))
    data['egpm_dominance_ema_after'] = (data.groupby([identity])
                                        ['egpm_dominance_ratio']
                                        .transform(lambda x: x.ewm(halflife=9, ignore_na=True).mean()))
    data['opp_egpm_dominance_ema_before'] = (data.groupby([identity])
                                             ['opp_egpm_dominance_ratio']
                                             .transform(lambda x: x.ewm(halflife=9, ignore_na=True)
                                                        .mean().shift().bfill()))
    data['opp_egpm_dominance_ema_after'] = (data.groupby([identity])
                                            ['opp_egpm_dominance_ratio']
                                            .transform(lambda x: x.ewm(halflife=9, ignore_na=True).mean()))

    data['egpm_dominance_win_perc'] = (data['egpm_dominance_ema_before'] /
                                       (data['egpm_dominance_ema_before'] +
                                        data['opp_egpm_dominance_ema_before']))
    data['egpm_dominance_diff'] = data['egpm_dominance_ema_before'] - data['opp_egpm_dominance_ema_before']

    return data


def dk_enrich(oe_data: pd.DataFrame, entity: str):
    """
    Calculate DraftKings point values for a player or team.

    Parameters
    ----------
    oe_data : pd.DataFrame
        A Pandas data frame containing cleaned Oracle's Elixir data.
    entity : str
        Calculate DK points using Team scoring or Player scoring. Value must be 'team' or 'player'.

    Returns
    -------
    A Pandas dataframe containing all of the original data from the cleaned
    OE input plus an added column containing DraftKings points for each game.
    """
    if entity == 'team':
        oe_data['dkpoints'] = (oe_data['towers'] +
                               (2 * oe_data['dragons']) +
                               (3 * oe_data['barons']) +
                               (2 * oe_data['firstblood']) +
                               (np.where(oe_data['result'] > 0, 2, 0)) +
                               (np.where((oe_data['gamelength'] / 60) < 30,
                                         2, 0)))
    elif entity == 'player':
        oe_data['dkpoints'] = ((3 * oe_data['kills']) +
                               (2 * oe_data['assists']) +
                               (-1 * oe_data['deaths']) +
                               (0.02 * oe_data['total cs']) +
                               (np.where((oe_data['kills'] > 10)
                                         | (oe_data['assists'] > 10), 2, 0)))
    else:
        raise ValueError("Must define either player or team scoring.")
    return oe_data


def enrich_ema_statistics(oe_data: pd.DataFrame, entity: str):
    oe_data["kda"] = (oe_data["kills"] + oe_data["assists"]) / oe_data["deaths"]
    if entity == 'team':
        identity = 'teamid'
        columns = ['kills', 'deaths', 'assists', 'earned gpm', 'gamelength',
                   'firstblood', 'dragons', 'barons', 'towers',
                   'goldat15', 'xpat15', 'csat15', 'golddiffat15', 'xpdiffat15', 'csdiffat15', 'dkpoints', 'kda', ]
    elif entity == 'player':
        identity = 'playerid'
        columns = ['kills', 'deaths', 'assists', 'total cs', 'earned gpm', 'earnedgoldshare', 'gamelength',
                   'goldat15', 'xpat15', 'csat15', 'assistsat15', 'deathsat15',
                   'golddiffat15', 'xpdiffat15', 'csdiffat15', 'dkpoints', 'kda']
    else:
        raise ValueError("Entity must be either team or player.")

    for col in columns:
        oe_data[f'{col}_ema_after'] = (oe_data.groupby([identity])[col]
                                       .transform(lambda x: x.ewm(halflife=5, ignore_na=True).mean()))
        oe_data[f'{col}_ema_before'] = (oe_data.groupby([identity])[col]
                                        .transform(lambda x: x.ewm(halflife=9, ignore_na=True).mean().shift().bfill()))

    return oe_data
