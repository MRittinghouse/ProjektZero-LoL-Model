from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import pandas as pd
from settings import BASE_DIR
import os


def create_ml_model_and_calculate_logloss(player_df):

    team_df = player_df.groupby(['gameid', 'teamid'])[['trueskill_mu', 'opponent_mu']].sum().reset_index()

    team_df = pd.merge(team_df, player_df[['gameid', 'teamid', 'result']], how='left', on=['gameid', 'teamid'])
    team_df['mu_difference'] = team_df['trueskill_mu']  - team_df['opponent_mu']
    y = team_df['result']
    X = team_df[['mu_difference']]

    model = LogisticRegression()
    model.fit(X, y)

    prob = model.predict_proba(X)[:,1]
    lr = log_loss(y, prob)

    return lr

if __name__ == '__main__':
    data_dir = os.path.join(BASE_DIR, "data")
    player_data_full_path = os.path.join(data_dir, "player_data")
    player_df = pd.read_pickle(player_data_full_path)
    create_ml_model_and_calculate_logloss(player_df)