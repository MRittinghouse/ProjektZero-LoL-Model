# -*- coding: utf-8 -*-
"""
The ProjektZero League of Legends Model

Please read the readme on the GitHub page at:
    https://github.com/MRittinghouse/ProjektZero-LoL-Model
    
Make sure to edit the "configurations.py" file to ensure that you are using
the correct variables for your environment before you run this! 
"""
import datetime
import lolmodeling
import configurations as conf
import numpy as np
import oracleselixir as oe
import os
import pandas as pd
import sys

def main(workingdir=conf.workingdir, 
         regions=conf.regions, 
         matches=conf.matches, 
         team_replacements=conf.team_replacements,
         player_replacements=conf.player_replacements,
         model_csvs=conf.csv,
         model_validations=conf.validate,
         ):
    start = datetime.datetime.now()
    current_year = datetime.date.today().year
    years = [str(current_year), str(current_year - 1)]
    
    # Data Stewardship
    if not os.path.exists(f'{workingdir}'):
        os.makedirs(f'{workingdir}')
    if not os.path.exists(f'{workingdir}\\Predictions'):
        os.makedirs(f'{workingdir}\\Predictions')
    if not os.path.exists(f'{workingdir}\\RawData'):
        os.makedirs(f'{workingdir}\\RawData')
    if not os.path.exists(f'{workingdir}\\ModelValidation'):
        os.makedirs(f'{workingdir}\\ModelValidation')
    if not os.path.exists(f'{workingdir}\\ModelOutputs'):
        os.makedirs(f'{workingdir}\\ModelOutputs')
    
    # Data Imports
    data = oe.download_data(workingdir, years=years, delete=True)
    
    if team_replacements:
        data['team'] = data['team'].replace(team_replacements)
    
    team_data = oe.clean_data(data, split_on='team')
    player_data = oe.clean_data(data, split_on='player')
    
    # Get Last Roster
    starting = oe.get_last_roster(player_data, regions)
    if player_replacements:
        new_starting = [player_replacements.get(
            item) if player_replacements.get(
                item) else item for item in starting]
        starting = new_starting
    
    #Calculate 2021 Team Elo
    current_team_elos, teameloacc = lolmodeling.current_year_elo(team_data, 
                                                     leagues=regions, 
                                                     directory=workingdir, 
                                                     csv=model_csvs, 
                                                     validate=model_validations)
    
    # Calculate "Player-Based" Team Elo
    player_elos, pleloacc = lolmodeling.player_based_elo(player_data, 
                                   players=starting, leagues=regions, 
                                   directory=workingdir, csv=model_csvs,
                                   validate=model_validations)
    
    #Calculate 2020/2021 "Player-Based" Team TrueSkill
    trueskill, egpmdata, tsacc = lolmodeling.team_trueskill(player_data, 
                                   leagues=regions, directory=workingdir, 
                                   csv=model_csvs, validate=model_validations)
    
    # EWM Model
    ewm = lolmodeling.ewm_modeling(team_data, workingdir, csv=False, leagues=regions)
    
    # EGPM Model
    team_egpm, egpmacc = lolmodeling.egpm_model(egpmdata, directory=workingdir, 
                                    leagues=regions, csv=True, validate=True)
    
    # Merge Elo Calculations
    metrics = current_team_elos.merge(
        player_elos, how='inner', on='id').reset_index()
    
    # Generate Summary Data Frame / Report
    summaryCols = ['kills', 'deaths', 'assists', 'earned gpm', 'gamelength', 
                   'team kpm', 'ckpm']
    team_data = team_data[team_data.league.isin(regions)].copy()
    team_data[summaryCols] = team_data[summaryCols].apply(pd.to_numeric, errors='coerce')
    team_data = team_data.groupby('team')[summaryCols].agg(
        [lolmodeling.q10, 'mean', lolmodeling.std, lolmodeling.q90])
    team_data.columns = team_data.columns.droplevel(1)
    metrics = metrics.merge(team_data, how='inner', left_on='id', 
                            right_on='team').reset_index()
    metrics.columns = ['level_0', 'index', 'id', 'league', 'last_played_date', 
                  'elo_current_year', 'player_elo', 
                  'kills_10', 'kills_avg', 'kills_std', 'kills_90',
                  'deaths_10', 'deaths_avg', 'deaths_std', 'deaths_90',
                  'assists_10', 'assists_avg', 'assists_std', 'assists_90',
                  'earnedgold_10', 'earnedgold_avg', 'earnedgold_std', 'earnedgold_90',
                  'time_10', 'time_avg', 'time_std', 'time_90', 
                  'team_kpm_10', 'team_kpm_avg', 'team_kpm_std', 'team_kpm_90',
                  'ckpm_10', 'ckpm_avg', 'ckpm_std', 'ckpm_90']
    metrics = metrics.drop(['level_0', 'index'], axis=1)
    metrics = metrics.merge(trueskill, how='inner', left_on='id', 
                            right_on='team').reset_index()
    metrics = metrics.merge(ewm, how='inner', left_on='id', 
                            right_on='team').reset_index()
    metrics = metrics.drop(['level_0', 'index', 'gameid', 'date', 
                            'team_x', 'team_y'], axis=1)
    metrics = metrics.merge(team_egpm, how='inner', left_on='id', 
                            right_on='team').reset_index()
    metrics = metrics.drop(columns=['index', 'gameid', 'date', 'team', 
                                    'result', 'blue_team_result'], axis=1)

    metrics.to_csv(f'{workingdir}\\Predictions\\stat_report.csv', index=False)
    
    # Output Upcoming Match Estimates
    estimates = lolmodeling.provide_win(metrics, 'Elo', 
                                    'elo_current_year', matches)
    estimates = estimates.rename(
        columns={'blue_win': 'team_elo_winperc', 
                 'elo_diff': 'team_elo_diff'})

    player_based_estimates = lolmodeling.provide_win(metrics, 'Elo', 
                                                     'player_elo', matches)
    player_based_estimates = player_based_estimates.rename(
        columns={'blue_win': 'player_elo_winperc', 
                 'elo_diff': 'player_elo_diff'})

    ts_estimates = lolmodeling.provide_win(metrics, 'TrueSkill', 
                                           None, matches)
    ts_estimates = ts_estimates.rename(columns={'blue_win': 'trueskill_winperc', 
                                                'elo_diff': 'mu_diff'})
    
    egpm_estimates = lolmodeling.provide_win(metrics, 'EGPM', None, matches)
    egpm_estimates = egpm_estimates.rename(columns={'blue_win': 'egpm_winperc',
                                                    'elo_diff': 'egpm_diff'})

    predictions = estimates.merge(
        player_based_estimates, on=['blue_team', 'red_team']).merge(
            ts_estimates, on=['blue_team', 'red_team']).merge(
                egpm_estimates, on=['blue_team', 'red_team'])
    pred_cols = ['team_elo_winperc', 'player_elo_winperc', 
                 'trueskill_winperc', 'egpm_winperc']
    sumaccuracy = teameloacc + pleloacc + tsacc + egpmacc
    predictions['blue_win_pct'] = ((predictions['team_elo_winperc'] * (teameloacc/sumaccuracy)) +
                                 (predictions['player_elo_winperc'] * (pleloacc/sumaccuracy)) +
                                 (predictions['trueskill_winperc'] * (tsacc/sumaccuracy)) +
                                 (predictions['egpm_winperc'] * (egpmacc/sumaccuracy)))
    
    predictions['pred_dev'] = predictions[pred_cols].std(axis=1)
    
    predictions['wgt_consensus'] = ((np.where(predictions['team_elo_winperc'] > 0.5, 1, 0) * (teameloacc/sumaccuracy)) +
                                    (np.where(predictions['player_elo_winperc'] > 0.5, 1, 0) * (pleloacc/sumaccuracy)) + 
                                    (np.where(predictions['trueskill_winperc'] > 0.5, 1, 0) * (tsacc/sumaccuracy)) +
                                    (np.where(predictions['egpm_winperc'] > 0.5, 1, 0) * (egpmacc/sumaccuracy)))
            
    predictions.to_csv(f'{workingdir}\\Predictions\\predictions.csv', index=False)
    print(f'{regions} Predictions: \n')
    print(predictions[['blue_team', 'red_team', 'blue_win_pct', 'pred_dev']])
    end = datetime.datetime.now()
    print(f'\nThis was computed using data up to {start} Eastern Time')
    print(f'Script Completed in {end - start}')

    
if __name__ in ('__builtins__', '__main__'):
    # If you want to customize the main function parameters, do it here! 
    main()
