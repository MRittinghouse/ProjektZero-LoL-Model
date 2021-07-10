# -*- coding: utf-8 -*-
"""
League of Legends Modeling TOols

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

Example:
    # Housekeeping
    import oracleselixir as oe
    import lolmodeling
    import pandas as pd
    
    # Data Import
    data = oe.download_data(dir, years, delete)
    team_data = oe.clean_data(data, split_on='team')
    
    # EWM Model
    ewm = lolmodeling.ewm_modeling(team_data, workingdir, 
                                   csv=False, leagues=regions)
"""
# Housekeeping
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm 
import seaborn as sns
from sklearn.metrics import log_loss
import sys
import trueskill

# Function Library
def std(x):
    """
    This just returns the standard deviation of X, calculated by Numpy.

    Parameters
    ----------
    x : float/int
        A number.

    Returns
    -------
    float
        Float value containing the standard deviation of X.
    """
    return np.std(x)

def q10 (x): 
    """
    This will return the 10th percentile of a Numpy series.
    """ 
    return np.percentile(x, q=10)

def q90(x):
    """
    This will return the 90th percentile of a Numpy series.
    """
    return np.percentile(x, q=90)

def elo_calculator(df, entity, start_elo=1200, k=25):
    """
    Calculates elo from Pandas dataframe
    
    Attributes:
        df (DataFrame): must contain winning team ['team'] 
                        and losing opponent ['opponent'] in order by game date
        entity: str ("team" or "player")
             The entity to calculate elo upon (team or player)
        start_elo (optional): identifies the starting Elo (default=1000)
        k (optional): provides the k-factor for the calcuation (default=100)
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
    def get_elo(team):
        return elo_results.get(team, start_elo)
    
    
    def update_elo(winner, loser):
        winner_elo = get_elo(winner)
        elo_winner_before.append(winner_elo)
        loser_elo = get_elo(loser)
        elo_loser_before.append(loser_elo)
        result_expected.append(1 if winner_elo > loser_elo else 0)
        expected_win = expected_result(winner_elo, loser_elo)
        change_in_elo = k * (1-expected_win)
        winner_elo += change_in_elo
        loser_elo -= change_in_elo
        elo_results[winner] = winner_elo
        elo_results[loser] = loser_elo
        elo_winner_after.append(winner_elo)
        elo_loser_after.append(loser_elo)
        elo_expected.append(expected_win)


    def expected_result(elo_a, elo_b):
        expect_a = 1.0/(1+10**((elo_b - elo_a)/400))
        return expect_a


    if entity not in ['team', 'player']:
        raise ValueError('Entity must be either PLAYER or TEAM.')   
    
    df.apply(lambda row: update_elo(row[entity], row['opponent']), axis=1)

    elo_dataframe = pd.DataFrame({'gameid': df['gameid'],
                                  'date': df['date'],
                                  'win_team_ref': df['team'],
                                  'position': df['position'],
                                  'winner': df[entity],
                                  'winning_elo_before': elo_winner_before,
                                  'winning_elo_after': elo_winner_after,
                                  'win_perc': elo_expected,
                                  'expected_result': result_expected,
                                  'lose_team_ref': df['opposing_team'],
                                  'loser': df['opponent'],
                                  'losing_elo_before': elo_loser_before,
                                  'losing_elo_after': elo_loser_after
                                  })
    return elo_dataframe


def current_year_elo(df, leagues, directory, csv, validate):
    """
    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing OE team data. 
    leagues : str or list of strs
        A string or list of strings containing leagues (e.g. ["LCS", "LEC"])
        Only teams from these Leagues will be in the final output.
    directory : str
        A string containing the filepath to the working directory.
        (e.g. 'C:\\Users\\ProjektStation\\Documents\\OraclesElixir\\')
    csv : boolean
        If True, will write a csv to working directory
    validate : boolean
        If True, will render a graph with precision/recall validation metrics.

    Returns
    -------
    A Pandas dataframe containing the latest elo scores for the teams in the 
        league specified within the leagues parameter. 
    A .csv file in the directory matching the dataframe, if specified by csv.
    A graph containing validation metrics for elo-based predictions if graph.
    """
    maxyear = df.date.max().year
    df = df[df['date'] > f'{maxyear}-01-01'] # Current Year Only
    df = df.sort_values('date')
    df = df[df['result'] == 1]
    eloData = elo_calculator(df, entity='team', start_elo=1200, k=28)
    
    # Conditional Handling For Leagues Input
    if isinstance(leagues, str):
        leagues = [leagues]
    
    # Merge Things Back Together
    df = eloData.merge(df, how='inner', left_on=['gameid', 'date', 'winner'], 
                       right_on=['gameid', 'date', 'team']).reset_index(drop=True)
    if csv:
        df.to_csv(f'{directory}teamelo_currentyear.csv', index=False)
    
    # Elo Validation Formula
    if validate:
        grf = sns.jointplot(data=df, x='winning_elo_before', 
                            y='losing_elo_before', hue='expected_result')
        plt.title('Current Year Team Elo', loc='right', y=1.1)
        correct = len(df[df['expected_result']==df['result']])/len(df)
        logloss = log_loss(df['result'], df['win_perc'], labels=[0,1])
        
        grf.ax_joint.text(df['winning_elo_before'].mean(), 
                          df['losing_elo_before'].max(), 
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')
        grf.savefig(f'{directory}CurrentYearTeamElo_Validation.png', dpi=300, 
                    format='png')
        plt.show()
        plt.clf()
    
    # Subset Selected Leagues Only
    if leagues:
        df = df[df['league'].isin(leagues)].reset_index(drop=True)
        
    # Combine Data Frames To Generate Output
    df1 = df[['winner', 'league', 'date', 'winning_elo_after']]
    df1 = df1.rename(columns={'winner': 'id', 
                              'date': 'last_played_date',
                              'winning_elo_after': 'elo_current_year'})
    df2 = df[['loser', 'league', 'date', 'losing_elo_after']]
    df2 = df2.rename(columns={'loser': 'id', 
                              'date': 'last_played_date',
                              'losing_elo_after': 'elo_current_year'})
    teamelo = pd.concat([df1, df2], ignore_index=True)
    
    # Elo Visualization
    for league in leagues:
        league_teamelo = teamelo[teamelo['league'] == league]
        pltx = sns.lineplot(x='last_played_date', y='elo_current_year', hue='id',
                               data=league_teamelo, palette='dark')
        pltx.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        pltx.tick_params(axis='x', labelrotation=40)
        pltx.figure.savefig(f'{directory}{league}Divergence_CurrentYearElo.png', 
                            dpi=300, format='png', bbox_inches='tight')
        plt.show()
        plt.clf()
    
    # Get Each Team's Most Recent Game / Elo Rankings
    teamelo = teamelo.sort_values(['id', 'last_played_date']).drop_duplicates(
        'id', keep='last', ignore_index=True)
    
    # Generate Outputs
    teamelo = teamelo.sort_values(['elo_current_year'], ascending=False, 
                                  ignore_index=True)

    return teamelo, correct


def player_based_elo(df, leagues, players, directory, csv, validate):
    """
    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing OE player data. 
    leagues : list
        A list of strings containing leagues (e.g. ["LCS", "LEC"])
        Only teams from these Leagues will be in the final output.
    players : list
        A list of strings containing the names of players that will be 
        starting this week. Used to consider only the starting roster.
        If None, all players that have played for that team will be considered.
    directory : str
        A string containing the filepath to the working directory.
        (e.g. 'C:\\Users\\ProjektStation\\Documents\\OraclesElixir\\')
    csv : boolean
        If True, will write a csv to working directory
    validate : boolean
        If True, will render a graph with precision/recall validation metrics.

    Returns
    -------
    A Pandas dataframe containing the latest elo scores for the TEAMS in the 
        league specified within the leagues parameter. 
    A .csv file in the directory matching the dataframe, if specified by csv.
    A graph containing validation metrics for elo-based predictions if graph.
    """       
    df = df.sort_values('date')
    df = df[df['result'] == 1]
    
    # Conditional Handling For Leagues Input
    if isinstance(leagues, str):
        leagues = [leagues]
    
    eloData = elo_calculator(df, entity='player', start_elo=1200, k=28)
    
    # Merge Things Back Together
    df = eloData.merge(df, how='inner', left_on=['gameid', 'date', 'winner'], 
                       right_on=['gameid', 'date', 'player']).reset_index(drop=True)
    maxyear = df.date.max().year
    df = df[df['date'] > f'{maxyear}-01-01'] # Current Year Only
    if csv:
        df.to_csv(f'{directory}playerbasedelo_player.csv', index=False)
    
    # Elo Validation Formula
    if validate:
        validation = df.groupby(['gameid', 'team', 'result']).agg({'winning_elo_before': 'mean', 
                                                         'losing_elo_before': 'mean',
                                                         'win_perc': 'mean'})
        validation.columns = ['winning_elo_before', 'losing_elo_before', 'win_perc']
        validation = validation.reset_index()
        validation['expected_result'] = np.where(validation['win_perc'] >= 0.5, 1, 0)
        
        grf = sns.jointplot(data=validation, x='winning_elo_before', 
                            y='losing_elo_before', hue='expected_result')
        plt.title('2-Year Player Elo', loc='right', y=1.1)     
        correct = len(df[df['expected_result']==df['result']])/len(df)
        logloss = log_loss(df['result'], df['win_perc'], labels=[0,1])
        
        grf.ax_joint.text(df['winning_elo_before'].min()*1.02, 
                          (df['losing_elo_before'].max()*0.975), 
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')
        grf.savefig(f'{directory}PlayerBasedElo_Validation.png', dpi=300, 
                    format='png')
        plt.show()
        plt.clf()
  
    # Subset Selected Leagues & Players Only    
    if leagues:
        df = df[df['league'].isin(leagues)].reset_index()
    if players:
        df = df[df['player'].isin(players)].reset_index(drop=True)

    # Combine Data Frames To Generate Output
    df1 = df[['win_team_ref', 'winner', 'date', 'winning_elo_after']].copy()
    df1 = df1.rename(columns={'win_team_ref': 'team',
                              'winner': 'id', 
                              'date': 'last_played_date',
                              'winning_elo_after': 'player_elo'})
    df2 = df[['lose_team_ref', 'loser', 'date', 'losing_elo_after']].copy()
    df2 = df2.rename(columns={'lose_team_ref': 'team',
                              'loser': 'id', 
                              'date': 'last_played_date',
                              'losing_elo_after': 'player_elo'})
    playerelo = pd.concat([df1, df2], ignore_index=True)
    playerelo = playerelo.sort_values(['id', 'last_played_date']).drop_duplicates(
        'id', keep='last', ignore_index=True)
    
    # Aggregate Individual Players Into Teams
    playerelo = playerelo.sort_values(
        ['player_elo'], ascending=False, ignore_index=True)
    team_elo_playerbased = playerelo.groupby('team').agg(['mean']).reset_index()
    team_elo_playerbased.columns = team_elo_playerbased.columns.droplevel(1)
    team_elo_playerbased = team_elo_playerbased.sort_values(
        ['player_elo'], ascending=False, ignore_index=True)
    team_elo_playerbased = team_elo_playerbased.rename(columns={'team': 'id'})
    if csv:
        team_elo_playerbased.to_csv(f'{directory}playerbasedelo_team.csv', 
                                    index=False)

    return team_elo_playerbased, correct


def team_trueskill(df, leagues, directory, csv, validate):
    """
    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing OE player data. 
    leagues : list
        A list of strings containing leagues (e.g. ["LCS", "LEC"])
        Only teams from these Leagues will be in the final output.
    directory : str
        A string containing the filepath to the working directory.
        (e.g. 'C:\\Users\\ProjektStation\\Documents\\OraclesElixir\\')
    csv : boolean
        If True, will write a csv to working directory
    validate : boolean
        If True, will render a graph with precision/recall validation metrics.

    Returns
    -------
    A Pandas dataframe containing the latest TrueSkill scores for the TEAMS 
        in the league specified within the leagues parameter. 
    A .csv file in the directory matching the dataframe, if specified by csv.
    A graph containing validation metrics for TS-based predictions if graph.
    """
    # Data Setup
    df = df.sort_values(by=['date', 'player'])
    input_data = df[['gameid', 'date', 'league', 'player', 'side', 'team', 
                     'position', 'result', 'earned gpm', 'ckpm', 'team kpm']]
    earned_gold = input_data.groupby(
        ['gameid', 'team'])['earned gpm'].sum().reset_index()
    earned_gold = earned_gold.rename(columns={'earned gpm': 'team_egpm'})
    input_data = pd.merge(input_data, earned_gold, on=['gameid', 'team'])
    input_data = input_data.rename(columns={'team kpm': 'team_kpm'})
    input_data[['team_egpm', 'team_kpm', 
                'ckpm']] = input_data[['team_egpm', 'team_kpm', 
                                       'ckpm']].fillna(0)
        
    # Conditional Handling For Leagues Input
    if isinstance(leagues, str):
        leagues = [leagues]
    
    # Initialize TrueSkill Player Ratings Dictionary
    ts = trueskill.TrueSkill(draw_probability=0.0)
    player_ratings_dict = dict()
    for i in input_data['player'].unique():
        player_ratings_dict[i] = ts.create_rating()
        
    def setup_match(df):
        df = df.sort_values(['league', 'date', 'gameid', 
                             'side', 'position']).reset_index()
        data = df[['player', 'date', 'result', 'team', 'league', 'ckpm', 
                   'gameid', 'team_egpm', 'team_kpm']].values
        actions = int(len(data)/10)
        pointer = 0
        outputframe = []
        for a in range(actions):
            if pointer == 0:
                ind = a
            else:
                ind = (a * 10)
            matchframe = [data[ind, 6], #gameid
                          data[ind, 1], #date
                          data[ind, 4], #league
                          data[ind, 5], #ckpm
                          data[ind, 7], #blue earned gpm
                          data[ind, 8], #blue kpm
                          data[ind, 3], #blue team
                          data[ind+4, 0], #blue top
                          data[ind+1, 0], #blue jng
                          data[ind+2, 0], #blue mid
                          data[ind, 0], # blue bot
                          data[ind+3, 0], #blue sup
                          data[ind, 2], #blue result
                          data[ind+5, 3], # red team
                          data[ind+5, 7], # red earned gpm
                          data[ind+5, 8], # red kpm
                          data[ind+9, 0], # red top
                          data[ind+6, 0], # red jng
                          data[ind+7, 0], # red mid
                          data[ind+5, 0], # red bot
                          data[ind+8, 0]]  # red sup 
            outputframe.append(matchframe)
            pointer += 1
            
        return outputframe

    columns = ['gameid', 'date', 'league', 'ckpm', 'blue_earned_gpm', 'blue_kpm',
               'blue_team', 'blue_top_name', 'blue_jng_name', 'blue_mid_name', 
               'blue_bot_name', 'blue_sup_name', 'blue_team_result', 
               'red_team', 'red_earned_gpm', 'red_kpm', 'red_top_name', 
               'red_jng_name', 'red_mid_name', 'red_bot_name', 'red_sup_name']
                    
    lcs_rating = pd.DataFrame(setup_match(input_data), columns=columns)
    
    analyzed_gameids = {}
    
    def win_probability(team1, team2, trueskill_global_env):
        delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
        sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
        size = len(team1) + len(team2)
        denom = math.sqrt(size * (trueskill_global_env.beta ** 2 ) + sum_sigma)
        return trueskill_global_env.cdf(delta_mu / denom)
    
    def update_trueskill(rating_dict, gameid_dict, gameid, blue_team_result,
                         blue_top_name, blue_jng_name, blue_mid_name, 
                         blue_bot_name, blue_sup_name, red_top_name, 
                         red_jng_name, red_mid_name, red_bot_name, 
                         red_sup_name):
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
        blue_team_win_prob = win_probability(rating_groups[0], rating_groups[1], ts)
        
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
            # for ranks 0 is winner 
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
        
        # update the rating dictionary
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
        
        # determine if use old data because of repeat
        if gameid in gameid_dict:
            return gameid_dict[gameid]
        else:
            gameid_dict[gameid] = ts_update
            return ts_update

    lcs_rating[['blue_team_win_prob','blue_mu','blue_sigma','red_mu','red_sigma',
                'blue_top_mu','blue_top_sigma','blue_jng_mu','blue_jng_sigma',
                'blue_mid_mu','blue_mid_sigma','blue_bot_mu','blue_bot_sigma',
                'blue_sup_mu','blue_sup_sigma','red_top_mu','red_top_sigma',
                'red_jng_mu','red_jng_sigma','red_mid_mu','red_mid_sigma',
                'red_bot_mu','red_bot_sigma','red_sup_mu',
                'red_sup_sigma']] = lcs_rating.apply(
                    lambda row: update_trueskill(
                        player_ratings_dict, analyzed_gameids, row['gameid'], 
                        row['blue_team_result'], row['blue_top_name'], 
                        row['blue_jng_name'], row['blue_mid_name'], 
                        row['blue_bot_name'], row['blue_sup_name'],
                        row['red_top_name'], row['red_jng_name'], 
                        row['red_mid_name'], row['red_bot_name'], 
                        row['red_sup_name']), axis=1)
    lcs_rating['blue_expected_result'] = np.where(
        lcs_rating['blue_team_win_prob'] >= 0.5, 1, 0)
    egpmdata = lcs_rating.copy()
    
    if csv:
        lcs_rating.to_csv(f'{directory}team_trueskill.csv', index=False)
        
    # Elo Validation Formula
    if validate:
        maxyear = lcs_rating.date.max().year
        lcs_rating = lcs_rating[lcs_rating['date'] > f'{maxyear}-01-01'].copy()
        lcs_rating['winner_mu'] = np.where(lcs_rating['blue_team_result']==1, 
                                           lcs_rating['blue_mu'], 
                                           lcs_rating['red_mu'])
        lcs_rating['loser_mu'] = np.where(lcs_rating['blue_team_result']==0, 
                                          lcs_rating['blue_mu'], 
                                          lcs_rating['red_mu'])      
        grf = sns.jointplot(data=lcs_rating, x='winner_mu', 
                            y='loser_mu', hue='blue_team_result')
        plt.title('2-Year Player TrueSkill', loc='right', y=1.1)
        
        correct = len(lcs_rating[lcs_rating['blue_expected_result']==
                                 lcs_rating['blue_team_result']])/len(lcs_rating)
        logloss = log_loss(lcs_rating['blue_team_result'], 
                           lcs_rating['blue_team_win_prob'], labels=[0,1])
        
        grf.ax_joint.text(lcs_rating['winner_mu'].min()*1.05, 
                          (lcs_rating['loser_mu'].max()*0.975), 
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')
        grf.savefig(f'{directory}TrueSkill_Validation.png', dpi=300, format='png')
        plt.show()
        plt.clf()
        
    # Rack 'em and Stack 'em
    if leagues:
        lcs_rating = lcs_rating[lcs_rating['league'].isin(leagues)].reset_index()
        
    blue_mu = ['blue_top_mu', 'blue_jng_mu', 'blue_mid_mu', 
               'blue_bot_mu', 'blue_sup_mu']
    blue_sigma = ['blue_top_sigma', 'blue_jng_sigma', 
                  'blue_mid_sigma', 'blue_bot_sigma', 'blue_sup_sigma']
    red_mu = ['red_top_mu', 'red_jng_mu', 'red_mid_mu', 
              'red_bot_mu', 'red_sup_mu']
    red_sigma = ['red_top_sigma', 'red_jng_sigma', 'red_mid_sigma', 
                 'red_bot_sigma', 'red_sup_sigma']
    blue = lcs_rating[['gameid', 'date', 'blue_team']].copy()
    blue['blue_sum_mu'] = lcs_rating[blue_mu].sum(axis=1)
    blue['blue_sum_sigma'] = lcs_rating[blue_sigma].sum(axis=1)
    blue = blue.rename(columns={'blue_team': 'team', 'blue_sum_mu': 'sum_mu', 
                                'blue_sum_sigma': 'sum_sigma'})
    red = lcs_rating[['gameid', 'date', 'red_team']].copy()
    red['red_sum_mu'] = lcs_rating[red_mu].sum(axis=1)
    red['red_sum_sigma'] = lcs_rating[red_sigma].sum(axis=1)
    red = red.rename(columns={'red_team': 'team', 'red_sum_mu': 'sum_mu', 
                              'red_sum_sigma': 'sum_sigma'})
    output = pd.concat([blue, red], ignore_index=True)
    output = output.sort_values(['team', 'date']).drop_duplicates(
        'team', keep='last', ignore_index=True)
    output = output.sort_values(['sum_mu'])
    if csv:
        output.to_csv(f'{directory}trueskill_output.csv', index=False)

    return output, egpmdata, correct


def ewm_modeling(df, directory, csv, leagues):
    """
    This function generates an Exponentially Weighted Mean (EWM) model
    to provide context on win rates based on side (red/blue) for teams.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing Team Data as from oe_data import split on team.
    directory : str
        A string containing the filepath to the working directory.
        (e.g. 'C:\\Users\\ProjektStation\\Documents\\OraclesElixir\\')
    csv : boolean
        If True, will write a csv to working directory
    leagues : str or list of strings
        A string or list of strings containing leagues (e.g. ["LCS", "LEC"])
        Only teams from these Leagues will be in the final output.

    Returns
    -------
    output : DataFrame
        A DataFrame containing the EWM model output for each team.

    """
    df['result'] = df['result'].astype(float)
    
    redside = df[df['side']=='Red'].copy()
    redside['red_side_ema'] = redside.groupby(
        ['team'])['result'].transform(lambda x: x.ewm(halflife=5).mean())
    
    blueside = df[df['side']=='Blue'].copy()
    blueside['blue_side_ema'] = blueside.groupby(
        ['team'])['result'].transform(lambda x: x.ewm(halflife=5).mean())
    
    merged = pd.concat([blueside, redside], ignore_index=True)
    
    # Conditional Handling For Leagues Input
    if isinstance(leagues, str):
        leagues = [leagues]
    
    if leagues:
        merged = merged[merged['league'].isin(leagues)].reset_index()
    
    teams = list(merged.team.unique())
    merged = merged.sort_values(['team', 'date'])
    dataset = []
    for t in teams:
        #t, red, blue
        temp = merged[merged['team']==t]
        temp = temp.ffill(axis=0)
        red = temp['red_side_ema'].dropna().iloc[-1]
        blue = temp['blue_side_ema'].dropna().iloc[-1]
        record = {'team': t, 'blue_side_ema': blue, 'red_side_ema': red}
        dataset.append(record)

    output = pd.DataFrame(dataset)
    if csv:
        output.to_csv(f'{directory}side_ewm_data.csv', index=False)
    return output


def egpm_model(data, directory, leagues, csv, validate):
    """
    This function grabs a secondary output of the TrueSkill function
    to calculate a model based on Earned GPM with TrueSkill factored in.

    Parameters
    ----------
    data : DataFrame
        The EGPM data output as provided by Team TrueSkill
    leagues : string or list of str
        The leagues of interest to keep.
    csv : Boolean
        True yields a .csv of results
    validate : Boolean
        True yields a validation graph with evaluation metrics.

    Returns
    -------
    DataFrame with Earned Gold Per Minute vs Opponent Team Strength model.
    """
    # Conditional Handling For Leagues Input
    if isinstance(leagues, str):
        leagues = [leagues]
        
    # Data Setup
    blue_mu = ['blue_top_mu', 'blue_jng_mu', 'blue_mid_mu', 'blue_bot_mu',
                 'blue_sup_mu']
    red_mu = ['red_top_mu', 'red_jng_mu', 'red_mid_mu', 'red_bot_mu',
                 'red_sup_mu']
    data['blue_sum_mu'] = data[blue_mu].sum(axis=1)
    data['red_sum_mu'] = data[red_mu].sum(axis=1)
    data['blue_dominance_ratio'] = (data['blue_earned_gpm'] / 
                                    (data['red_sum_mu'] / 
                                     data['blue_sum_mu']))
    data['red_dominance_ratio'] = (data['red_earned_gpm'] / 
                                   (data['blue_sum_mu'] / 
                                    data['red_sum_mu']))
    data['blue_dominance_ema'] = data.groupby(
        ['blue_team'])['blue_dominance_ratio'].transform(
            lambda x: x.ewm(halflife=9).mean())
    data['red_dominance_ema'] = data.groupby(
        ['red_team'])['red_dominance_ratio'].transform(
            lambda x: x.ewm(halflife=9).mean())
    data['blue_expected_result'] = np.where(data['blue_dominance_ema'] >= 
                                            data['red_dominance_ema'], 1, 0)
    data['blue_win_perc'] = (data['blue_dominance_ema']/
                             (data['blue_dominance_ema']+
                              data['red_dominance_ema']))
    
    if validate:
        maxyear = data.date.max().year
        data = data[data['date'] > f'{maxyear}-01-01'] # Current Year Only
        grf = sns.jointplot(data=data, x='blue_dominance_ema', 
                            y='red_dominance_ema', hue='blue_team_result')
        plt.title('TrueSkill-Normalized EGPM', loc='right', y=1.1)
            
        correct = len(data[data['blue_expected_result']==
                           data['blue_team_result']])/len(data)
        logloss = log_loss(data['blue_team_result'], data['blue_win_perc'], 
                           labels=[0,1])
        
        grf.ax_joint.text(data['blue_dominance_ema'].min()*1.025, 
                          data['red_dominance_ema'].max()*0.975, 
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')
        grf.savefig(f'{directory}EGPM_Dominance_Validation.png', dpi=300, format='png')
        plt.show()
        plt.clf()
    
    if leagues:
        data = data[data['league'].isin(leagues)].reset_index()
    
    blue = data[['gameid', 'date', 'blue_team', 'blue_dominance_ema', 
                 'blue_team_result']].copy()
    blue = blue.rename(
        columns={'blue_team': 'team', 
                 'blue_dominance_ema': 'dominance_ema',
                 'blue_team_result': 'result'})
    
    red = data[['gameid', 'date', 'red_team', 'red_dominance_ema', 
                'blue_team_result']].copy()
    red['red_team_result'] = 1 - red['blue_team_result']
    red = red.rename(
        columns={'red_team': 'team', 
                 'red_dominance_ema': 'dominance_ema',
                 'red_team_result': 'result'})
    merged = pd.concat([blue, red], axis=0)
    if csv:
        merged.to_csv(f'{directory}EGPM_RawOutput.csv', index=False)
    output = merged.sort_values(['team', 'date']).drop_duplicates(
        'team', keep='last', ignore_index=True)

    if csv:
        output.to_csv(f'{directory}EGPM_Dominance.csv', index=False)
    return output, correct


def provide_win(df, variant, elo_field, matchups):
    """
    Calculates win percentage for a given dict of matchups.
    
    df : DataFrame 
        DataFrame containing team names (in col: "id") and elo values.
    variant: str
        String of type of prediction (options: 'Elo', 'TrueSkill')
    elo_field : str
        String of the column name containing Elo value to use for predictions.
        Used in Elo prediction variants only.
    matchups : Dict containing list of strings 
        Dictionary of upcoming matches, team names must match df
        Format of {'game1': ['blue_team_name', 'red_team_name']}
        
    Output is a dataframe containing predictions and probabilities for matchups.
    """
    teamblue = []
    teamred = []
    bluewin = []
    elo_diff = []
        
    if variant == 'Elo':
        def get_elo(df, elo_field, team):
            query = df.query(f'id=="{team}"')
            return query[elo_field].iloc[0]
        
        def expected_result(elo_a, elo_b):
            expect_a = 1.0/(1+10**((elo_b - elo_a)/400))
            return expect_a
        
        for game, v in matchups.items():
            team1 = v[0]
            team2 = v[1]
            
            # These Try Blocks Flag Team Name Mismatches 
            # Between The Schedule API and OE Data
            # If this hits, update the "misnomers" dictionary
            # In the "upcoming schedule" function
            try:
                team1elo = get_elo(df, elo_field, team1)
            except:
                print(team1)
                sys.exit(1)
            try:
                team2elo = get_elo(df, elo_field, team2)
            except:
                print(team2)
                sys.exit(1)
                
            winperc = expected_result(team1elo, team2elo)
            teamblue.append(team1)
            teamred.append(team2)
            bluewin.append(winperc)
            elo_diff.append(team1elo - team2elo)
            
        results = pd.DataFrame({'blue_team': teamblue,
                                'red_team': teamred,
                                'blue_win': bluewin,
                                'elo_diff': elo_diff})
    elif variant == 'TrueSkill':            
        for game, v in matchups.items():
            team1 = v[0]
            team2 = v[1]
            deltamu = (df[df.id==f'{team1}'].sum_mu.iloc[0] - 
                       df[df.id==f'{team2}'].sum_mu.iloc[0])
            sumsig = ((df[df.id==f'{team1}'].sum_sigma.iloc[0] ** 2) + 
                      (df[df.id==f'{team2}'].sum_sigma.iloc[0]))
            denom = math.sqrt(10 * (4.1666666667 ** 2) + sumsig)
            winperc = norm.cdf(deltamu / denom)
            teamblue.append(team1)
            teamred.append(team2)
            bluewin.append(winperc)
            elo_diff.append(deltamu)
            
        results = pd.DataFrame({'blue_team': teamblue,
                                'red_team': teamred,
                                'blue_win': bluewin,
                                'elo_diff': elo_diff})
        
    elif variant == 'EGPM':
        for game, v in matchups.items():
            team1 = v[0]
            team2 = v[1]
            blue_dom = df[df.id==f'{team1}']['dominance_ema'].iloc[0]
            red_dom = df[df.id==f'{team2}']['dominance_ema'].iloc[0]
            winperc = (blue_dom / (blue_dom + red_dom))
            teamblue.append(team1)
            teamred.append(team2)
            bluewin.append(winperc)
            elo_diff.append((blue_dom - red_dom))
            
        results = pd.DataFrame({'blue_team': teamblue,
                                'red_team': teamred,
                                'blue_win': bluewin,
                                'elo_diff': elo_diff})
            
    return results


def Bo3(t1name, t1odds, t2name, t2odds):
    """
    Parameters
    ----------
    t1name : str
        Team 1 Name.
    t1odds : float
        Team 1 Odds.
    t2name : str
        Team 2 Name.
    t2odds : float
        Team 2 Odds.

    Returns
    -------
    A string containing predictions and outputs for a Best of 3 series.
    Yields both the overall win likelihood, and game score probabilities.
    This is performed by taking the model output odds and applying basic
        multiple probability formulae.
    """
    # Team 1 2/0:
    t1_20 = (t1odds * t1odds)
        
    # Team 1 2/1
    t1_21 = (t1odds * t2odds * t1odds) + (t2odds * t1odds * t1odds)
    
    # Team 2 2/0
    t2_20 = (t2odds * t2odds)
    
    # Team 2 2/1
    t2_21 = (t2odds * t1odds * t2odds) + (t1odds * t2odds * t2odds)
    
    # Final Outputs
    doublecheck = t1_20 + t1_21 + t2_20 + t2_21
    assert round(doublecheck, 5) == 1.0
    
    output = f'''
Overall Likelihood Of {t1name} To Win Series: {((t1_20 + t1_21) * 100):.2f}%
    Probability {t1name} wins 2/0: {(t1_20*100):.2f}%
    Probability {t1name} wins 2/1: {(t1_21*100):.2f}%

Overall Likelihood Of {t2name} To Win Series: {((t2_20 + t2_21) * 100):.2f}%
    Probability {t2name} wins 2/0: {(t2_20*100):.2f}%
    Probability {t2name} wins 2/1: {(t2_21*100):.2f}%'''
    return output


def Bo5(t1name, t1odds, t2name, t2odds):
    """
    Parameters
    ----------
    t1name : str
        Team 1 Name.
    t1odds : float
        Team 1 Odds.
    t2name : str
        Team 2 Name.
    t2odds : float
        Team 2 Odds.

    Returns
    -------
    A string containing predictions and outputs for a Best of 5 series.
    Yields both the overall win likelihood, and game score probabilities.
    This is performed by taking the model output odds and applying basic
        multiple probability formulae.
    """
    
    # Team 1 3/0:
    t1_30 = (t1odds * t1odds * t1odds)
        
    # Team 1 3/1
    t1_31 = ((t1odds * t1odds * t2odds * t1odds) +
            (t2odds * t1odds * t1odds * t1odds) +
            (t1odds * t2odds * t1odds * t1odds))
    
    # Team 1 3/2
    t1_32 = ((t2odds * t2odds * t1odds * t1odds * t1odds) +
             (t2odds * t1odds * t1odds * t2odds * t1odds) + 
             (t2odds * t1odds * t2odds * t1odds * t1odds) + 
             (t1odds * t2odds * t2odds * t1odds * t1odds) +
             (t1odds * t1odds * t2odds * t2odds * t1odds) + 
             (t1odds * t2odds * t1odds * t2odds * t1odds))
    
    # Team 2 3/0
    t2_30 = (t2odds * t2odds * t2odds)
    
    # Team 2 3/1
    t2_31 = ((t2odds * t2odds * t1odds * t2odds) +
            (t1odds * t2odds * t2odds * t2odds) +
            (t2odds * t1odds * t2odds * t2odds))
    
    # Team 2 3/2
    t2_32 = ((t1odds * t1odds * t2odds * t2odds * t2odds) +
             (t1odds * t2odds * t2odds * t1odds * t2odds) + 
             (t1odds * t2odds * t1odds * t2odds * t2odds) + 
             (t2odds * t1odds * t1odds * t2odds * t2odds) +
             (t2odds * t2odds * t1odds * t1odds * t2odds) + 
             (t2odds * t1odds * t2odds * t1odds * t2odds))
    
    # Final Outputs
    doublecheck = t1_30 + t1_31 + t1_32 + t2_30 + t2_31 + t2_32
    assert round(doublecheck, 5) == 1.0
    
    output = f'''
Overall Likelihood Of {t1name} To Win Series: {((t1_30 + t1_31 + t1_32) * 100):.2f}%
    Probability {t1name} wins 3/0: {(t1_30*100):.2f}%
    Probability {t1name} wins 3/1: {(t1_31*100):.2f}%
    Probability {t1name} wins 3/2: {(t1_32*100):.2f}%
            
Overall Likelihood Of {t2name} To Win Series: {((t2_30 + t2_31 + t2_32) * 100):.2f}%
    Probability {t2name} wins 3/0: {(t2_30*100):.2f}%
    Probability {t2name} wins 3/1: {(t2_31*100):.2f}%
    Probability {t2name} wins 3/2: {(t2_32*100):.2f}%'''
    return output