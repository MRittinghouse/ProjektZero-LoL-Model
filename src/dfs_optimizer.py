# -*- coding: utf-8 -*-
"""
LoL DFS Optimizer

This script is intended to serve as an optimizer for daily fantasy sports 
games for League of Legends. 

This is for academic purposes only, and has no expectation or guarantee of 
performance. Nothing in this script or its outputs constitutes financial advice.
The user assumes all risk for any usage of this script. 

This is an early rendition of this script, and is intended to be expanded upon.

"""
# Housekeeping
import json
import random
import pandas as pd
import sys


# Function Library
def prep_dk_csv(dk, fade_list):
    # Subset/Format Data
    dk['Name'] = dk['Name'].str.strip()
    dk = dk[~dk['Name'].isin(fade_list)]

    # Define Daily Matches And Favorites
    raw_match_list = list(set(dk['Game Info']))
    match_list = []

    for match in raw_match_list:
        sep = ' '
        match = match.split(sep, 1)[0]
        match = match.replace("@", '" , "')
        match = '["' + match + '"]'
        match_list.append(match)

    # Generate Match Evaluation String
    i = 1
    match_string = "{"
    for match in match_list:
        string = "\"game" + str(i) + "\": " + match + ","
        match_string = match_string + string
        i += 1

    match_string = match_string[:-1] + "}"
    match_string = json.loads(match_string)

    return match_string


def optimizer(df: pd.DataFrame, salary_cap=1500000, max_team_size=3):
    """
    Please note, this is a fairly naive optimizer. 
    However, it can be customized. You only need five core columns. 
    If you have a custom model you want to plug in, simply overwrite the 
    "pts" column with your custom expected point values. 
    
    Additionally, E1 Fantasy does not allow for .csv files to be directly
    downloaded from their site. You may need to build your own .csv file. 
    
    Parameters
    ----------
    df: pd.DataFrame
        Pandas DataFrame containing DraftKings rosters and costs for the slate of interest.
    filepath : str
        The filepath to a .csv file containing columns ['role', 'player', 
        'team', 'salary', 'pts'] columns as shown on E1 Fantasy 
        (www.fantasy.esportsone.com/).
        Please note that the data in the 'role' column should be in all caps.
    salary_cap : int
        The total maximum amount of salary that you have to spend on players.
    max_team_size : int
        The maximum number of players permitted from the same team.

    Returns
    -------
    dict
        An dict object containing the optimal DFS roster using E1 rules.

    """
    df = df[['player', 'role', 'team', 'salary', 'pts']]

    top = df[df['role'] == 'TOP']
    jng = df[df['role'] == 'JNG']
    mid = df[df['role'] == 'MID']
    adc = df[df['role'] == 'BOT']
    sup = df[df['role'] == 'SUP']
    team = df[df['role'] == 'TEAM']

    best_roster = {'Top': '', 'Jng': '', 'Mid': '', 'Bot': '', 'Sup': '', 'Team': '', 'Cost': int(0), 'Score': float(0)}

    for adc_ind, adc_row in adc.iterrows():
        for mid_ind, mid_row in mid.iterrows():
            for jng_ind, jng_row in jng.iterrows():
                for top_ind, top_row in top.iterrows():
                    for sup_ind, sup_row in sup.iterrows():
                        for team_ind, team_row in team.iterrows():
                            ind_cost = (top.at[top_ind, 'salary'] +
                                        jng.at[jng_ind, 'salary'] +
                                        mid.at[mid_ind, 'salary'] +
                                        adc.at[adc_ind, 'salary'] +
                                        sup.at[sup_ind, 'salary'] +
                                        team.at[team_ind, 'salary'])
                            ind_score = (top.at[top_ind, 'pts'] +
                                         jng.at[jng_ind, 'pts'] +
                                         mid.at[mid_ind, 'pts'] +
                                         adc.at[adc_ind, 'pts'] +
                                         sup.at[sup_ind, 'pts'] +
                                         team.at[team_ind, 'pts'])

                            # Remove rosters over salary cap or under point threshold
                            if ind_cost > salary_cap:
                                continue
                            if ind_score < best_roster['pts']:
                                continue

                            top_team = top.at[top_ind, 'team']
                            top_name = top.at[top_ind, 'player']
                            jng_team = jng.at[jng_ind, 'team']
                            jng_name = jng.at[jng_ind, 'player']
                            mid_team = mid.at[mid_ind, 'team']
                            mid_name = mid.at[mid_ind, 'player']
                            adc_team = adc.at[adc_ind, 'team']
                            adc_name = adc.at[adc_ind, 'player']
                            sup_team = sup.at[sup_ind, 'team']
                            sup_name = sup.at[sup_ind, 'player']
                            team_team = team.at[team_ind, 'team']
                            team_name = team.at[team_ind, 'player']

                            # Remove rosters with more than 3 items from same team
                            teams = [top_team, jng_team, mid_team, adc_team, sup_team, team_team]
                            flag = ""
                            for i in set(teams):
                                count = teams.count(i)
                                if count > max_team_size:
                                    flag = "Yes"
                            if flag == "Yes":
                                continue

                            best_roster = {'Top': str(top_team + ' ' + top_name),
                                           'Jng': str(jng_team + ' ' + jng_name),
                                           'Mid': str(mid_team + ' ' + mid_name),
                                           'ADC': str(adc_team + ' ' + adc_name),
                                           'Sup': str(sup_team + ' ' + sup_name),
                                           'Team': str(team_team + ' ' + team_name),
                                           'Cost': int(ind_cost),
                                           'Score': float(ind_score)}

    del best_roster['Score']

    if best_roster == '''{'Top': '', 'Jng': '', 'Mid': '', 'ADC': '', 'Sup': '', 'Team': '', 'Cost': 0}''':
        raise Exception("No mathematically possible rosters for this stack.")
    else:
        return best_roster
