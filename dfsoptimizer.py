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
                        
# Roster Generator
def optimizer(filepath, salarycap=1500000, maxteamsize=3):
    """
    Please note, this is a fairly naive optimizer. 
    However, it can be customized. You only need five core columns. 
    If you have a custom model you want to plug in, simply overwrite the 
    "pts" column with your custom expected point values. 
    
    Additionally, E1 Fantasy does not allow for .csv files to be directly
    downloaded from their site. You may need to build your own .csv file. 
    
    Parameters
    ----------
    filepath : str
        The filepath to a .csv file containing columns ['role', 'player', 
        'team', 'salary', 'pts'] columns as shown on E1 Fantasy 
        (www.fantasy.esportsone.com/).
        Please note that the data in the 'role' column should be in all caps.
    salarycap : int
        The total maximum amount of salary that you have to spend on players.
    maxteamsize : int
        The maximum number of players permitted from the same team.

    Returns
    -------
    dict
        An dict object containing the optimal DFS roster using E1 rules.

    """
    df = pd.read_csv(f'{filepath}')
    df = df[['player', 'role', 'team', 'salary', 'pts']]

    top  = df[df['role'] == 'TOP']
    jng  = df[df['role'] == 'JNG']
    mid  = df[df['role'] == 'MID']
    adc  = df[df['role'] == 'BOT']
    sup  = df[df['role'] == 'SUP']
    team = df[df['role'] == 'TEAM']
    
    
    bestroster = {'Top':'', 
         'Jng':'',
         'Mid':'',
         'Bot':'',
         'Sup':'', 
         'Team':'',
         'Cost':int(0),
         'Score':float(0)}
    
    for adcind, adcrow in adc.iterrows():
        for midind, midrow in mid.iterrows():
            for jngind, jngrow in jng.iterrows():
                for topind, toprow in top.iterrows():
                    for supind, suprow in sup.iterrows():
                        for teamind, teamrow in team.iterrows():
                            indcost = (top.at[topind,'salary']+
                                       jng.at[jngind,'salary']+
                                       mid.at[midind,'salary']+
                                       adc.at[adcind,'salary']+
                                       sup.at[supind,'salary']+
                                       team.at[teamind,'salary'])
                            indscore = 0.00
                            indscore = (top.at[topind,'pts']+
                                        jng.at[jngind,'pts']+
                                        mid.at[midind,'pts']+
                                        adc.at[adcind,'pts']+
                                        sup.at[supind,'pts']+
                                        team.at[teamind,'pts'])
                                
                            ## Remove rosters over salary cap or under point threshold
                            if indcost>salarycap:
                                continue
                            if indscore<bestroster['pts']:
                                continue
                                        
                            topteam = top.at[topind,'team']
                            topname = top.at[topind,'player']
                            jngteam = jng.at[jngind,'team']
                            jngname = jng.at[jngind,'player']
                            midteam = mid.at[midind,'team']
                            midname = mid.at[midind,'player']
                            adcteam = adc.at[adcind,'team']
                            adcname = adc.at[adcind,'player']
                            supteam = sup.at[supind,'team']
                            supname = sup.at[supind,'player']
                            teamteam = team.at[teamind,'team']
                            teamname = team.at[teamind,'player']
                                        
                            ## Remove rosters with more than 3 items from same team
                            teams = [topteam, jngteam, midteam, adcteam, supteam, teamteam]
                            flag = ""
                            for i in set(teams):
                                count = teams.count(i)
                                if count > maxteamsize:
                                    flag = "Yes"
                            if flag == "Yes":
                                continue
                                            
                            bestroster = {'Top':str(topteam+' '+topname), 
                                          'Jng':str(jngteam+' '+jngname),
                                          'Mid':str(midteam+' '+midname),
                                          'ADC':str(adcteam+' '+adcname),
                                          'Sup':str(supteam+' '+supname), 
                                          'Team':str(teamteam+' '+teamname),
                                          'Cost':int(indcost),
                                          'Score':float(indscore)}
                                
    del bestroster['Score']
    
    if bestroster == '''{'Top': '', 'Jng': '', 'Mid': '', 'ADC': '', 'Sup': '', 'Team': '', 'Cost': 0}''':
        raise Exception("No mathematically possible rosters for this stack.")
    else:
        return bestroster
