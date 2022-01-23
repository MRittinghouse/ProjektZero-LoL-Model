# Enrich Team TrueSkill
import src.lol_modeling as lol
import pandas as pd
# Clean/Format Data
# Define time frame for analytics
import datetime as dt
import os
import src.oracles_elixir as oe
import pickle
from settings import BASE_DIR
from machine_learning_model import create_ml_model_and_calculate_logloss

current_year = dt.date.today().year
years = [str(current_year), str(current_year - 1), str(current_year - 2)]

SAVE_PICKLE = False

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

data_dir = os.path.join(BASE_DIR, "data")
player_data_full_path = os.path.join(data_dir, "player_data")

min_logloss = float('inf')
min_days_sigma_multiplier = float('inf')
min_sigma_negative_bonus_per_match = float('inf')

for days_sigma_multiplier in [0, 0.1, 0.15, 0.25]:
    for sigma_negative_bonus_per_match in [0.1, 0.2, 0.3]:

        player_data, _, _, _ = lol.trueskill_model(players, teams,
                                                   days_sigma_multiplier=days_sigma_multiplier,
                                                   sigma_negative_bonus_per_match=sigma_negative_bonus_per_match)

        if SAVE_PICKLE:
            player_data.to_pickle(player_data_full_path)

        logloss = create_ml_model_and_calculate_logloss(player_data)
        if logloss < min_logloss:
            min_logloss = logloss
            min_days_sigma_multiplier = days_sigma_multiplier
            min_sigma_negative_bonus_per_match = sigma_negative_bonus_per_match

        print(days_sigma_multiplier, sigma_negative_bonus_per_match, logloss)

print("Best", logloss, min_days_sigma_multiplier, min_sigma_negative_bonus_per_match)
