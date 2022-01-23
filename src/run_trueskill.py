# Enrich Team TrueSkill
import src.lol_modeling as lol
import pandas as pd
# Clean/Format Data
# Define time frame for analytics
import datetime as dt
import src.oracles_elixir as oe
current_year = dt.date.today().year
years = [str(current_year), str(current_year - 1), str(current_year - 2)]

# Download Data
data = oe.download_data(years=years)

# Remove Buggy Matches (both red/blue team listed as same team, invalid for elo/TrueSkill)
invalid_games = ['NA1/3754345055', 'NA1/3754344502',
                 'ESPORTSTMNT02/1890835', 'NA1/3669212337',
                 'NA1/3669211958', 'ESPORTSTMNT02/1890848']
data = data[~data.gameid.isin(invalid_games)].copy()

# Clean/Format Data
teams = oe.clean_data(data, split_on='team')
players = oe.clean_data(data, split_on='player')
player_data, team_data, last_match_dict, ts_lookup = lol.trueskill_model(players, teams)