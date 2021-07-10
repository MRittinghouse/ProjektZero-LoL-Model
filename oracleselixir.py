# -*- coding: utf-8 -*-
"""
Oracle's Elixir Data Interface

This script is designed to connect to Tim Sevenhuysen's Oracle's Elixir site
    to pull down and format data. 
    
Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.

This code is intended to be imported into other Python analytics projects

Example:
    import oracleselixir as oe
    import pandas as pd
    
    data = oe.download_data(dir, years, delete)
"""
# Housekeeping
import datetime
import io
import json
 #from MyCredentials import key, url # This is a custom file I wrote that contains my credentials.
import numpy as np
import os
import pandas as pd
import requests
import sys

# Function Definitions
def download_data(directory, years, delete):
    """
    Parameters
    ----------
    directory : str
        A string containing the filepath to the working directory.
        (e.g. 'C:\\Users\\ProjektStation\\Documents\\OraclesElixir\\')
    year : str or list
        A string or list of strings containing years (e.g. ["2019", "2020"])
    delete : boolean
        A boolean (True/False) value. 
        If True, will delete files in directory upon download of new data.

    Returns
    -------
    A Pandas dataframe containing the most recent Oracle's Elixir data 
    for the years provided by the year parameter.
    A .csv file in the directory, with the most recent data, if downloaded.
    """
    # Defining Time Variables
    current_date = datetime.date.today()
    today = current_date.strftime('%Y%m%d')
    yesterday = current_date - datetime.timedelta(days = 1)
    yesterday = yesterday.strftime('%Y%m%d')
    
    # Other Variables
    url = ('https://oracleselixir-downloadable-match-data.'
        's3-us-west-2.amazonaws.com/')
    oe_data = pd.DataFrame()
    
    # Conditional Handling For Years
    if isinstance(years, str):
        years = [years]
    
    # Dynamic Data Import
    for year in years:
        file = f'{year}_LoL_esports_match_data_from_OraclesElixir_'
        current_files = [x for x in os.listdir(directory) if x.startswith(file)]
        
        if f'{file}{today}.csv' not in current_files:
             # If today's data not in Dir, optionally delete old versions
             if delete:
                 for f in current_files:
                     print(f'DELETED: {f}')
                     os.remove(f'{directory}{f}')
                
             try:
                # Try To Grab File For Current Date
                filepath = f'{url}{file}{today}.csv'
                
                r = requests.get(filepath, allow_redirects=True)
                data = r.content.decode('utf8')
                data = pd.read_csv(io.StringIO(data), low_memory=False)
                data.to_csv(f'{directory}{file}{today}.csv', index=False)  
                print(f'DOWNLOADED: {file}{today}.csv')
             except:
                # Grab Yesterday's Data If Today's Does Not Exist
                filepath = f'{url}{file}{yesterday}.csv'
            
                r = requests.get(filepath, allow_redirects=True)
                data = r.content.decode('utf8')
                data = pd.read_csv(io.StringIO(data), low_memory=False)
                data.to_csv(f'{directory}{file}{yesterday}.csv', index=False)
                print(f'DOWNLOADED: {file}{yesterday}.csv')
        else:
            # Grab Local Data If It Exists
            data = pd.read_csv(f'{directory}{file}{today}.csv', 
                               low_memory=False)
            print(f'USING LOCAL {year} DATA')
        
        # Concatenate Data To Master Data Frame
        oe_data = pd.concat([oe_data, data], axis=0)
    
    return oe_data


def clean_data(oe_data, split_on):
    """
    Parameters
    ----------
    oe_data : DataFrame
        A Pandas data frame containing Oracle's Elixir data.
    split_on : 'team', 'player' or None 
        Subset data for Team data or Player data. None for all data.

    Returns
    -------
    A Pandas dataframe of formatted, subset Oracle's Elixir data matching 
    the parameters provided above. 
    The date column will be formatted appropriately as a datetime object.
    Only games with datacompletness = complete will be kept.
    Any games with 'unknown team' or 'unknown player' will be dropped. 
    Any games with null game ids will be dropped.
    """
    # Subset Columns and Define Column Data Types
    oe_data = oe_data[['date', 'gameid', 'datacompleteness', 'side', 
                       'position', 'league', 'player', 'team', 
                       'result', 'kills', 'deaths', 'assists', 'earned gpm', 
                       'gamelength', 'ckpm', 'team kpm']]
    oe_data = oe_data.astype({'date': 'datetime64'})
    
    # Keep Only "Complete" Games
    oe_data = oe_data[oe_data['datacompleteness'] == 'complete'].copy()
    oe_data = oe_data.drop(columns=['datacompleteness'], axis=1)
    
    # Remove Any Games With Null GameIDs
    oe_data['gameid'] = oe_data['gameid'].str.strip()
    oe_data['gameid'].replace('', np.nan, inplace=True)
    oe_data = oe_data[oe_data['gameid'].notna()]
    
    # Remove Any Records With Null Position Data
    oe_data['position'].replace('', np.nan, inplace=True)
    oe_data = oe_data[oe_data.position.notna()]
    
    # Split On Player/Team (and defining some related variables)
    if split_on == 'team':
        oe_data = oe_data[oe_data['position'] == 'team']
        cap = 2
        dropval = 'unknown team'
    elif split_on == 'player':
        oe_data = oe_data[oe_data['position'] != 'team'].copy()
        cap = 10
        dropval = 'unknown player'
    else:
        raise ValueError('Must split on either player or team.')
        
    # Remove Games That Don't Have Data For All Players
    counts = oe_data['gameid'].value_counts().to_frame()
    counts = counts[counts['gameid'] < cap]
    if len(counts) > 0:
        dropgames = counts.index.to_list()
        oe_data = oe_data[~oe_data.gameid.isin(dropgames)].copy()
        
    # Drop Games With "Unknown Player/Team" Lookup Failures
    dropgames = oe_data[oe_data[split_on] == dropval].copy()
    dropgames = dropgames['gameid'].unique()
    oe_data = oe_data[~oe_data.gameid.isin(dropgames)].copy().reset_index(
        drop=True)
    
    # Generate Opponent Column
    def _get_opponent(column, entity):
        """
        Parameters
        ----------
        column : Pandas Series
            Pandas DataFrame column representing entity data (see entity)
        entity : str
            'player' or 'team', entity to calculate opponent of.

        Returns
        -------
        opponent : Pandas Series
            The opponent of the entity in the column provided.

        """
        opponent = []
        if entity == 'player':
            gap = 5
        elif entity == 'team':
            gap = 1
        flag = 0
        for i, obj in enumerate(column):
            if flag < gap:
                opponent.append(column[i+gap])
                flag += 1
            elif flag >= gap and flag < (gap * 2):
                opponent.append(column[i-gap])
                flag += 1
            if flag >= (gap * 2):
                flag = 0
        return opponent

    oe_data = oe_data.sort_values(['league', 'date', 'gameid', 
                                   'side', 'position'])
    oe_data['opponent'] = _get_opponent(oe_data[split_on].to_list(), 
                                        split_on)
    
    if split_on == 'player':
        oe_data['opposing_team'] = _get_opponent(oe_data.team.to_list(), 
                                                 split_on)
    elif split_on == 'team':
        oe_data['opposing_team'] = oe_data['opponent']
    
    # Return Output
    return oe_data


def getLastRoster(playerdata, regions):
    """
    Parameters
    ----------
    playerdata : Pandas DataFrame
        Pandas DataFrame containing data subset to contain only player records.
    regions : str or list
        Leagues or Regions to keep

    Returns
    -------
    lastStarting : list
        List of players who most recently played with that player.

    """
    playerdata = playerdata[['league', 'date', 'team', 'position', 'player']]
    if regions:
        playerdata = playerdata[playerdata['league'].isin(regions)].reset_index()
    lastPlayed = playerdata.sort_values(
        ['date', 'team']).drop_duplicates(
            subset=['team'], keep='last', ignore_index=True)
    lastPlayed = lastPlayed['date'].tolist()
    lastPlayed = min(lastPlayed)
    lastStarting = playerdata[playerdata['date'] >= lastPlayed].reset_index()
    lastStarting = list(lastStarting.player.unique())
    return lastStarting


def _upcoming_schedule(leagues, days, url, key):
    """
    This function pings a currently private API endpoint 
    to grab a dictionary of upcoming matches.
    
    Please note, this API has not yet been released, so if you do not have 
    a valid key, do not try to use this function. 

    Parameters
    ----------
    leagues : str or list of strings
        A string or list of strings containing Leagues of interest 
        Valid Leagues are LCS, LEC, LCK, LPL.
        The API does not support all leagues, stick to the big four.
    days : int
        Number of days forward to search for matches (today + X days).
    url : str
        The URL of the API to pull data from.
    key : str
        The private key used to credential your access to the API. 

    Returns
    -------
    upcoming : dict
        A dictionary with the format of {ID: [blue, red]} for upcoming matches.
    """
    headers = {'x-api-key':f'{key}'}
    current_date = datetime.datetime.now()
    future_date = current_date + datetime.timedelta(days = days)
    tformat = '%Y-%m-%dT%H:%M:%S.%fZ'
    
    # Get API Response
    res = requests.get(url, headers=headers)
    
    # Get Schedule or Error Response
    if res.status_code == 200:
        schedule = json.loads(res.text)
    else:
        print(f'Error: {res.status_code}')
        sys.exit(1)
    
    # Conditional Handling For Leagues Input
    if isinstance(leagues, str):
        leagues = [leagues]
    
    # Filter Schedule of Interest
    league_schedule = list(filter(lambda game: game['league'] in leagues, 
                                  schedule))
    upcoming = list(filter(lambda game: 
                           datetime.datetime.strptime(game['startTime'], 
                                                tformat) <= future_date and 
                           datetime.datetime.strptime(game['startTime'], 
                                                tformat) >= current_date,
                           league_schedule))

    # Misnomers Format: (API Name) : (OE Name)
    # This will need to be regularly monitored and updated.
    # If you get key errors, update this dictionary.
    misnomers = {'Schalke 04': 'FC Schalke 04 Esports',
                 'Edward Gaming': 'EDward Gaming',
                 'kt Rolster': 'KT Rolster',
                 'TT': 'ThunderTalk Gaming',
                 'NONGSHIM REDFORCE': 'Nongshim RedForce',
                 'EXCEL': 'Excel Esports',
                 'Dignitas': 'Dignitas QNTMPAY'}
    
    def team_namer(teamname, misnomers):
        """
        Parameters
        ----------
        teamname : str
            Name of team from the data set.
        misnomers : dict
            Dictionary object containing misspellings/other names from the data.

        Returns
        -------
        str
            The correct team name.

        """
        if teamname in misnomers.keys():
            return misnomers[teamname]
        else:
            return teamname

    # If this block of code fails, there's a problem with the team names
    for i in upcoming:
        if i['team1Code'] == 'TBD' or i['team2Code'] == 'TBD':
            upcoming.remove(i)
    upcoming = map(lambda x: {x['matchId']: [team_namer(x['team1Name'], 
                                                        misnomers), 
                                             team_namer(x['team2Name'], 
                                                        misnomers)]}, 
                   upcoming)
    upcoming = dict(j for i in upcoming for j in i.items())
    
    return upcoming

def name_change(df, entity, legend):
    """
    This function is intended to replace values in the data in the event of 
    a player or team name change within the dataset.
    This is particularly useful when looking at multiple splits of data. 
    
    For example, if Dignitas renames to Dignitas QNTMPAY, use this function
    to rename all values of Dignitas to Dignitas QNTMPAY for consistency. 

    Parameters
    ----------
    df : Pandas DataFrame
        A Pandas DataFrame containing Oracles Elixir data.
    entity : str
        The name of the column to operate on (e.g. 'team', 'player')
    legend : dict
        A dictionary object of format {'old1': 'new1', 'old2':'new2'}

    Returns
    -------
    Pandas DataFrame
        An updated Pandas DataFrame
            with the values replaced according to the lookup legend provided.
    """
    
    output = df[entity].replace(legend)
    return output