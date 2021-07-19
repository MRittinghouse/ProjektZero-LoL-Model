# -*- coding: utf-8 -*-
"""
Configurations

These are the core variables that you need to execute the main script.
"""
# Housekeeping
import oracleselixir as oe

# Variable Definitions
"""
The working directory, or workingdir, represents the folder name  / filepath
where you want data to be saved. This will include all outputs, including the
raw data, graphs, model validation metrics, and spreadsheets. 

An example would be: 'C:\\Users\\YourMachineName\Documents\\OraclesElixir'
Note: Please don't end the path with \\, or things will break.
"""
workingdir = 'C:\\Users\\matth\Documents\\OraclesElixir'

"""
Regions represents which regions of gameplay / leagues you want to predict.
This value can be either a string or a list of strings.

Examples would be: 
    'LCS' 
    ['LCS', 'LEC', 'LPL', 'LCK']
"""
regions = ['LCS', 'LEC']

"""
Matches is intended to represent a dictionary object. 

There are two ways to pull this. The first is by using a function that requires
 you to have credentialed access to a currently-unreleased API. If you do not 
have access to that API, the next best bet is to navigate to lolesports.com
and manually build the dictionary object yourself here.  

If you need to manually build the dictionary, use the following format:
    {'game1': ['blue team', 'red team'],
     'game2': ['blue team', 'red team'],
     etc.}
Make sure that all spelling, spacing, and capitalization matches OraclesElixir. 
An example dictionary has been provided containing July 11th, 2021's LCS games.
    
If you are aware of any free, public APIs that can be used to pull down the
schedule for upcoming matches, please let me know as it would help a lot! 
"""
try:
    from MyCredentials import key, url
    days = 3 # Sets the time window. e.g: "all matches in the next 3 days"
    matches = oe._upcoming_schedule(regions, days, url, key)
except Exception as e:
    print(e)
    matches = {'game1': ['Team Liquid', 'Golden Guardians'], 
               'game2': ['Cloud9', 'Immortals'],
               'game3': ['100 Thieves', 'TSM'],
               'game4': ['Evil Geniuses', 'Cloud9'],
               'game5': ['Dignitas QNTMPAY', 'Counter Logic Gaming']}

"""
The team replacements variable is used to replace values in the data in the 
event of a team name change within the dataset.
This is particularly useful when looking at multiple splits of data. 
It's important to note that the default setting for the model is to use two
years of data, so you'll want to have corrections here for all major team 
name changes in the last two years of gameplay. 

The format is intended to be {'oldname1': 'newname1', 'oldname2': 'newname2'}
Be sure that the spelling, spacing, and capitalization match Oracle's Elixir.
"""
team_replacements = {'Dignitas': 'Dignitas QNTMPAY'}

"""
The player replacements variable used to document upcoming roster changes. 
The codebase in its current form uses a team's last fielded roster. 
If you are aware of an upcoming substitution, fill this dictionary out with 
the format {oldplayer1: newplayer1, oldplayer2: newplayer2, ...}. Be sure that
the spelling and capitalization are exactly as they are on Oracle's Elixir.
"""
player_replacements = {'Jenkins': 'Alphari'}

"""
CSV is a Boolean (True/False) variable that if True, renders CSV files.
This is useful if you want to dig into the individual models, or want to 
examine the data with a bit more detail. It's also really useful for 
troubleshooting bugs. I leave it set to True by default.
"""
csv = True

"""
Validate is a Boolean variable (True/False) that if True, renders model
validation graphs. These come with some model metrics such as log loss, as 
well as precision/recall in some cases. If you're a data scientist, or are 
interested in seeing more about the uncertainty / performance of the model,
validation is helpful. 

Recommended to leave enabled. 
"""
validate = True



