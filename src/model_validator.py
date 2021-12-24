# Housekeeping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import log_loss


def validate_team_elo(teams: pd.DataFrame, directory: Path):
    # Data Preparation
    teams['opp_team_elo'] = teams['team_elo_before'] - teams['team_elo_diff']
    teams['team_elo_expected_result'] = np.where(teams['team_elo_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Generate Graph
    grf = sns.jointplot(data=teams,
                        x='team_elo_before',
                        y='opp_team_elo',
                        hue='result')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['team_elo_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['team_elo_win_perc'], labels=[0, 1])
    grf.ax_joint.text(teams['team_elo_before'].mean(),
                      teams['opp_team_elo'].max(),
                      f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')

    # Format and Export
    grf.set_axis_labels("Blue Team Elo", "Red Team Elo")
    plt.title('Team Elo', loc='right', y=1.1)

    grf.savefig(directory.joinpath('TeamElo_Validation.png'), dpi=300, format='png')
    plt.show()
    plt.clf()

    return correct, logloss


def validate_player_elo(teams: pd.DataFrame, directory: Path):
    # Data Preparation
    teams['opp_player_elo'] = teams['player_elo_before'] - teams['player_elo_diff']
    teams['player_elo_expected_result'] = np.where(teams['player_elo_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Generate Graph
    grf = sns.jointplot(data=teams,
                        x='player_elo_before',
                        y='opp_player_elo',
                        hue='result')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['player_elo_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['player_elo_win_perc'], labels=[0, 1])
    grf.ax_joint.text(teams['player_elo_before'].mean(),
                      teams['opp_player_elo'].max(),
                      f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')

    # Format and Export
    grf.set_axis_labels("Blue Players' Elo", "Red Players' Elo")
    plt.title('Player Elo', loc='right', y=1.1)

    grf.savefig(directory.joinpath('PlayerElo_Validation.png'), dpi=300, format='png')
    plt.show()
    plt.clf()

    return correct, logloss


def validate_trueskill(teams: pd.DataFrame, directory: Path):
    # Data Preparation
    teams['trueskill_expected_result'] = np.where(teams['trueskill_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Generate Graph
    grf = sns.jointplot(data=teams,
                        x='trueskill_sum_mu',
                        y='opponent_sum_mu',
                        hue='result')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['trueskill_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['trueskill_win_perc'], labels=[0, 1])
    grf.ax_joint.text(teams['trueskill_sum_mu'].mean() - (teams['trueskill_sum_mu'].mean() * 0.25),
                      teams['opponent_sum_mu'].max(),
                      f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')

    # Format and Export
    grf.set_axis_labels("Blue Players' TrueSkill", "Red Players' TrueSkill")
    plt.title('TrueSkill', loc='right', y=1.1)

    grf.savefig(directory.joinpath('TrueSkill_Validation.png'), dpi=300, format='png')
    plt.show()
    plt.clf()

    return correct, logloss


def validate_egpm_dominance(teams: pd.DataFrame, directory: Path):
    # Data Preparation
    teams['egpm_dominance_win_perc'] = teams['egpm_dominance_win_perc'].fillna(0.5)
    teams['egpm_dominance_expected_result'] = np.where(teams['egpm_dominance_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Generate Graph
    grf = sns.jointplot(data=teams,
                        x='egpm_dominance_ema_before',
                        y='opp_egpm_dominance_ema_before',
                        hue='result')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['egpm_dominance_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['egpm_dominance_win_perc'], labels=[0, 1])
    grf.ax_joint.text(teams['egpm_dominance_ema_before'].mean() - (teams['egpm_dominance_ema_before'].mean() * 0.40),
                      teams['opp_egpm_dominance_ema_before'].max(),
                      f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')

    # Format and Export
    grf.set_axis_labels("Blue Players' EGPM Dominance", "Red Players' EGPM Dominance")
    plt.title('EGPM Dominance', loc='right', y=1.1)

    grf.savefig(directory.joinpath('EGPMDom_Validation.png'), dpi=300, format='png')
    plt.show()
    plt.clf()

    return correct, logloss


# def validate_side_ewm(teams: pd.DataFrame, players: pd.DataFrame):
# This is really hard to validate right now due to the data structure. Going to put a pin in this.
# return


def validate_ensemble_accuracy(teams: pd.DataFrame,
                               team_accuracy: float,
                               player_accuracy: float,
                               trueskill_accuracy: float,
                               egpm_dom_accuracy: float,
                               directory: Path):
    # Data Preparation
    teams['result'] = teams.result.astype('int32')
    sum_accuracy = team_accuracy + player_accuracy + trueskill_accuracy + egpm_dom_accuracy
    teams['ensemble_win_perc'] = ((teams['team_elo_win_perc'] * (team_accuracy / sum_accuracy)) +
                                  (teams['player_elo_win_perc'] * (player_accuracy / sum_accuracy)) +
                                  (teams['trueskill_win_perc'] * (trueskill_accuracy / sum_accuracy)) +
                                  (teams['egpm_dominance_win_perc'] * (egpm_dom_accuracy / sum_accuracy)))
    teams['opp_ensemble_win_perc'] = 1 - teams['ensemble_win_perc']
    teams['ensemble_expected_result'] = np.where(teams['ensemble_win_perc'] >= 0.5, 1, 0)

    # Generate Graph
    grf = sns.jointplot(data=teams,
                        x='ensemble_win_perc',
                        y='opp_ensemble_win_perc',
                        hue='result')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['ensemble_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['ensemble_win_perc'], labels=[0, 1])
    grf.ax_joint.text(teams['ensemble_win_perc'].mean() - (teams['ensemble_win_perc'].mean() * 0.45),
                      teams['opp_ensemble_win_perc'].max(),
                      f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}')

    # Format and Export
    grf.set_axis_labels("Blue Players' Ensemble Prediction", "Red Players' Ensemble Prediction")
    plt.title('Ensemble Model', loc='right', y=1.1)

    grf.savefig(directory.joinpath('EnsembleModel_Validation.png'), dpi=300, format='png')
    plt.show()
    plt.clf()

    return correct, logloss


# Main
def main():
    # Data Imports
    team_data = pd.read_csv(Path.cwd().parent.joinpath('data', 'interim', 'team_data.csv'))
    directory = Path.cwd().parent.joinpath('reports', 'figures')

    # Validation Individual Core Models
    te_acc, te_lls = validate_team_elo(team_data, directory)
    pe_acc, pe_lls = validate_player_elo(team_data, directory)
    ts_acc, ts_lls = validate_trueskill(team_data, directory)
    ed_acc, ed_lls = validate_egpm_dominance(team_data, directory)

    # Validate Ensemble Prediction
    validate_ensemble_accuracy(team_data, te_acc, pe_acc, ts_acc, ed_acc, directory)


if __name__ in ('__main__', '__builtin__', 'builtins'):
    main()
