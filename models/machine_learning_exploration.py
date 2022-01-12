"""
Machine Learning Exploration

Initial attempts at building the ensemble algorithm didn't quite perform as well as I wanted.
Here is an attempt at leveraging TPOT to create a more optimized machine learning pipeline.
"""
# Housekeeping
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Data Import
team_data = pd.read_csv(Path.cwd().parent.joinpath('data', 'interim', 'team_data.csv'))

# Data Subset
input_data = team_data[['league', 'side', 'result',
                        'team_elo_before', 'team_elo_diff', 'team_elo_win_perc',
                        'player_elo_before', 'player_elo_diff', 'player_elo_win_perc',
                        'trueskill_sum_mu', 'trueskill_sum_sigma', 'trueskill_diff', 'trueskill_win_perc',
                        'egpm_dominance_ema_before', 'egpm_dominance_diff', 'egpm_dominance_win_perc',
                        'blue_side_ema_before', 'red_side_ema_before',
                        'kills_ema_before', 'deaths_ema_before', 'assists_ema_before', 'earned gpm_ema_before',
                        'gamelength_ema_before', 'dragons_ema_before', 'barons_ema_before', 'towers_ema_before',
                        'csdiffat15_ema_before', 'golddiffat15_ema_before', 'xpdiffat15_ema_before',
                        'dkpoints_ema_before']].copy()
input_data.rename(columns={'result': 'target'}, inplace=True)

# Encode Leagues / Sides
league_enc = preprocessing.LabelEncoder()
input_data['league'] = league_enc.fit_transform(input_data['league'].copy())

side_enc = preprocessing.LabelEncoder()
input_data['side'] = side_enc.fit_transform(input_data['side'].copy())

# Initialize TPOT
output_directory = r'C:\Users\matth\Documents\GitHub\ProjektZero-LoL-Model\models'
pipeline_optimizer = TPOTClassifier(generations=100,
                                    population_size=60,
                                    subsample=0.50,
                                    n_jobs=1,
                                    max_eval_time_mins=15,
                                    cv=5,
                                    verbosity=3,
                                    memory='auto',
                                    # config_dict='TPOT light',
                                    periodic_checkpoint_folder=output_directory)

# Test/Train Split
X = input_data.drop(['target'], axis=1)
y = input_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

# May your CPU and RAM have mercy upon your soul.
pipeline_optimizer.fit(X_train, y_train)
pipeline_optimizer.export('tpot_pipeline.py')
print(pipeline_optimizer.score(X_test, y_test))
