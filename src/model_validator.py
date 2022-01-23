# Housekeeping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from sklearn.metrics import log_loss

sns.set_style("darkgrid")

# Variable Definitions
region_lookup = {"KR": ["CK", "KeSPA", "LAS", "LCK", "LCKC", "LCK CL"],
                 "CN": ["DC", "DCup", "NEST", "LDL", "LPL"],
                 "EU": ["BL", "BM", "CT", "DL", "EBL", "EU CS", "EUM", "GLL",
                        "HC", "HM", "HS", "LEC", "LFL", "LPLOL", "NEXO",
                        "NLC", "OTBLX", "PGN", "PRM", "SL", "UKLC", "UL"],
                 "NA": ["AOL", "BIG", "CU", "EGL", "GSG", "LCS", "LCSA",
                        "NA CS", "NASG", "NERD", "RCL", "UGP", "UPL"],
                 "BR": ["BRCC", "CBLOL", "CBLOLA"],
                 "TR": ["TCL", "TRA"],
                 "CIS": ["LCL", "CISC"],
                 "SEA": ["GPL", "LNL", "PCS"],
                 "VIET": ["VCS"],
                 "JPN": ["LJL", "LJLA", "LJLCS"],
                 "OCE": ["LCO", "OCS", "OPL"],
                 "LATAM": ["LHE", "LLA", "LMF", "LVP DDH"],
                 "INTL": ["IEM", "IWCI", "MSC", "MSI", "Riot", "WLDs"]}


def validate_team_elo(teams: pd.DataFrame, directory: Path, graph: bool):
    """
    Validate model performance of Team Elo model.

    Parameters
    ----------
    teams: pd.DataFrame
        DataFrame containing processed data, as output by the data_generator.py file
    directory: Path
        Filepath pointing to the reports/figures directory.
    graph: bool
        Boolean value indicating whether or not to generate the optional 300 dpi .png graph image.

    Returns
    -------
    [OPTIONAL] A 300 dpi .png image of a graph containing model validation metrics, in the directory specified.

    accuracy: float
        Variable describing the number of predictions that this model had correct.
    logloss: float
        Variable describing the log loss, as defined by sklearn.metrics, of the model.
    """
    # Data Preparation
    teams['opp_team_elo'] = teams['team_elo_before'] - teams['team_elo_diff']
    teams['team_elo_expected_result'] = np.where(teams['team_elo_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['team_elo_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['team_elo_win_perc'], labels=[0, 1])

    # Generate Graph
    if graph:
        grf = sns.jointplot(data=teams,
                            x='team_elo_before',
                            y='opp_team_elo',
                            hue='result')

        grf.ax_joint.text(teams['team_elo_before'].mean(),
                          teams['opp_team_elo'].max(),
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}',
                          bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        # Format and Export
        grf.set_axis_labels("Blue Team Elo", "Red Team Elo",
                            bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))
        plt.title('Team Elo', loc='right', y=1.1,
                  bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        grf.savefig(directory.joinpath('TeamElo_Validation.png'), dpi=300, format='png')
        plt.show()
        plt.clf()

    return correct, logloss


def validate_player_elo(teams: pd.DataFrame, directory: Path, graph: bool):
    """
    Validate model performance of Player Elo model.

    Parameters
    ----------
    teams: pd.DataFrame
        DataFrame containing processed data, as output by the data_generator.py file
    directory: Path
        Filepath pointing to the reports/figures directory.
    graph: bool
        Boolean value indicating whether or not to generate the optional 300 dpi .png graph image.

    Returns
    -------
    [OPTIONAL] A 300 dpi .png image of a graph containing model validation metrics, in the directory specified.

    accuracy: float
        Variable describing the number of predictions that this model had correct.
    logloss: float
        Variable describing the log loss, as defined by sklearn.metrics, of the model.
    """
    # Data Preparation
    teams['opp_player_elo'] = teams['player_elo_before'] - teams['player_elo_diff']
    teams['player_elo_expected_result'] = np.where(teams['player_elo_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['player_elo_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['player_elo_win_perc'], labels=[0, 1])

    # Generate Graph
    if graph:
        grf = sns.jointplot(data=teams,
                            x='player_elo_before',
                            y='opp_player_elo',
                            hue='result')

        grf.ax_joint.text(teams['player_elo_before'].mean(),
                          teams['opp_player_elo'].max(),
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}',
                          bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        # Format and Export
        grf.set_axis_labels("Blue Players' Elo", "Red Players' Elo",
                            bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))
        plt.title('Player Elo', loc='right', y=1.1,
                  bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        grf.savefig(directory.joinpath('PlayerElo_Validation.png'), dpi=300, format='png')
        plt.show()
        plt.clf()

    return correct, logloss


def validate_trueskill(teams: pd.DataFrame, directory: Path, graph: bool):
    """
    Validate model performance of Player-Based TrueSkill model.

    Parameters
    ----------
    teams: pd.DataFrame
        DataFrame containing processed data, as output by the data_generator.py file
    directory: Path
        Filepath pointing to the reports/figures directory.
    graph: bool
        Boolean value indicating whether or not to generate the optional 300 dpi .png graph image.

    Returns
    -------
    [OPTIONAL] A 300 dpi .png image of a graph containing model validation metrics, in the directory specified.

    accuracy: float
        Variable describing the number of predictions that this model had correct.
    logloss: float
        Variable describing the log loss, as defined by sklearn.metrics, of the model.
    """
    # Data Preparation
    teams['trueskill_expected_result'] = np.where(teams['trueskill_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['trueskill_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['trueskill_win_perc'], labels=[0, 1])

    # Generate Graph
    if graph:
        grf = sns.jointplot(data=teams,
                            x='trueskill_sum_mu',
                            y='opponent_sum_mu',
                            hue='result')

        grf.ax_joint.text((teams['trueskill_sum_mu'].mean() -
                          (teams['trueskill_sum_mu'].mean() * 0.25)),
                          teams['opponent_sum_mu'].max(),
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}',
                          bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        # Format and Export
        grf.set_axis_labels("Blue Players' TrueSkill", "Red Players' TrueSkill",
                            bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))
        plt.title('TrueSkill', loc='right', y=1.1,
                  bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        grf.savefig(directory.joinpath('TrueSkill_Validation.png'), dpi=300, format='png')
        plt.show()
        plt.clf()

    return correct, logloss


def validate_egpm_dominance(teams: pd.DataFrame, directory: Path, graph: bool):
    """
    Validate model performance of TrueSkill-Normalized EGPM Dominance model.

    Parameters
    ----------
    teams: pd.DataFrame
        DataFrame containing processed data, as output by the data_generator.py file
    directory: Path
        Filepath pointing to the reports/figures directory.
    graph: bool
        Boolean value indicating whether or not to generate the optional 300 dpi .png graph image.

    Returns
    -------
    A 300 dpi .png image of a graph containing model validation metrics, in the directory specified.

    accuracy: float
        Variable describing the number of predictions that this model had correct.
    logloss: float
        Variable describing the log loss, as defined by sklearn.metrics, of the model.
    """
    # Data Preparation
    teams['egpm_dominance_expected_result'] = np.where(teams['egpm_dominance_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['egpm_dominance_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['egpm_dominance_win_perc'], labels=[0, 1])

    # Generate Graph
    if graph:
        grf = sns.jointplot(data=teams,
                            x='egpm_dominance_ema_before',
                            y='opp_egpm_dominance_ema_before',
                            hue='result')

        grf.ax_joint.text((teams['egpm_dominance_ema_before'].min() +
                           (teams['egpm_dominance_ema_before'].mean() * 0.25)),
                          teams['opp_egpm_dominance_ema_before'].max(),
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}',
                          bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        # Format and Export
        grf.set_axis_labels("Blue Players' EGPM Dominance", "Red Players' EGPM Dominance",
                            bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))
        plt.title('EGPM Dominance', loc='right', y=1.1,
                  bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        grf.savefig(directory.joinpath('EGPMDom_Validation.png'), dpi=300, format='png')
        plt.show()
        plt.clf()

    return correct, logloss


def validate_side_ema(teams: pd.DataFrame, directory: Path, graph: bool):
    """
    Validate model performance of Side Win Rate Exponential Moving Average model.

    Parameters
    ----------
    teams: pd.DataFrame
        DataFrame containing processed data, as output by the data_generator.py file
    directory: Path
        Filepath pointing to the reports/figures directory.
    graph: bool
        Boolean value indicating whether or not to generate the optional 300 dpi .png graph image.

    Returns
    -------
    [OPTIONAL] A 300 dpi .png image of a graph containing model validation metrics, in the directory specified.

    accuracy: float
        Variable describing the number of predictions that this model had correct.
    logloss: float
        Variable describing the log loss, as defined by sklearn.metrics, of the model.
    """
    # Data Preparation
    teams['side_ema_expected_result'] = np.where(teams['side_ema_win_perc'] >= 0.5, 1, 0)
    teams['result'] = teams.result.astype('int32')

    # Label Accuracy and Log Loss
    correct = len(teams[teams['side_ema_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['side_ema_win_perc'], labels=[0, 1])

    # Generate Graph
    if graph:
        grf = sns.jointplot(data=teams,
                            x='side_ema_before',
                            y='opp_side_ema_before',
                            hue='result')

        grf.ax_joint.text(teams['side_ema_before'].mean(),
                          teams['opp_side_ema_before'].max(),
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}',
                          bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        # Format and Export
        grf.set_axis_labels("Blue Side EMA", "Red Side EMA",
                            bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))
        plt.title('Side Win% EMA', loc='right', y=1.1,
                  bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        grf.savefig(directory.joinpath('SideEMA_Validation.png'), dpi=300, format='png')
        plt.show()
        plt.clf()

    return correct, logloss


def validate_ensemble_accuracy(teams: pd.DataFrame,
                               team_accuracy: float,
                               player_accuracy: float,
                               trueskill_accuracy: float,
                               egpm_dom_accuracy: float,
                               side_ema_accuracy: float,
                               directory: Path,
                               graph: bool):
    """
    Weights each of the individual models into an ensemble based on their accuracy.
    The ensemble model is used to generate historical predictions.
    Once the predictions have been generated, the ensemble model is validated for its accuracy.

    Parameters
    ----------
    teams: pd.DataFrame
        DataFrame containing processed data, as output by the data_generator.py file.
    team_accuracy: float
        The accuracy of the Team Elo model as output by the validate_team_elo function.
    player_accuracy: float
        The accuracy of the Player Elo model as output by the validate_player_elo function.
    trueskill_accuracy: float
        The accuracy of the TrueSkill model as output by the validate_trueskill function.
    egpm_dom_accuracy: float
        The accuracy of the EGPM Dominance model as output by the validate_egpm_dominance function.
    side_ema_accuracy: float
        The accuracy of the Side Win EMA model as output by the validate_side_ema function.
    directory: Path
        Filepath pointing to the reports/figures directory.
    graph: bool
        Boolean value indicating whether or not to generate the optional 300 dpi .png graph image.

    Returns
    -------
    [OPTIONAL] A 300 dpi .png image of a graph containing model validation metrics, in the directory specified.

    accuracy: float
        Variable describing the number of predictions that this model had correct.
    logloss: float
        Variable describing the log loss, as defined by sklearn.metrics, of the model.
    """
    # Data Preparation
    teams['result'] = teams.result.astype('int32')
    sum_accuracy = team_accuracy + player_accuracy + egpm_dom_accuracy + side_ema_accuracy + trueskill_accuracy
    teams['ensemble_win_perc'] = ((teams['team_elo_win_perc'] * (team_accuracy / sum_accuracy)) +
                                  (teams['player_elo_win_perc'] * (player_accuracy / sum_accuracy)) +
                                  (teams['trueskill_win_perc'] * (trueskill_accuracy / sum_accuracy)) +
                                  (teams['egpm_dominance_win_perc'] * (egpm_dom_accuracy / sum_accuracy)) +
                                  (teams['side_ema_win_perc'] * (side_ema_accuracy / sum_accuracy)))
    teams['opp_ensemble_win_perc'] = 1 - teams['ensemble_win_perc']
    teams['ensemble_expected_result'] = np.where(teams['ensemble_win_perc'] >= 0.5, 1, 0)

    # Label Accuracy and Log Loss
    correct = len(teams[teams['ensemble_expected_result'] == teams['result']]) / len(teams)
    logloss = log_loss(teams['result'], teams['ensemble_win_perc'], labels=[0, 1])

    # Generate Graph
    if graph:
        grf = sns.jointplot(data=teams,
                            x='ensemble_win_perc',
                            xlim=(0, 1),
                            y='opp_ensemble_win_perc',
                            ylim=(0, 1),
                            hue='result')

        grf.ax_joint.text(0.15,
                          0.95,
                          f'Acc.: {correct:.4f} / Log Loss: {logloss:.4f}',
                          bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        # Format and Export
        grf.set_axis_labels("Blue Players' Ensemble Prediction", "Red Players' Ensemble Prediction",
                            bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))
        plt.title('Ensemble Model', loc='right', y=1.1,
                  bbox=dict(facecolor='grey', edgecolor='black', boxstyle='round'))

        grf.savefig(directory.joinpath('EnsembleModel_Validation.png'), dpi=300, format='png')
        plt.show()
        plt.clf()

    return correct, logloss


def generate_validation_metrics(graph: bool):
    """
    Generates validation metrics, and represents the primary data generation function for use in other scripts.

    Parameters
    ----------
    graph: bool
        Boolean value indicating whether or not to generate the optional 300 dpi .png graph image.

    Returns
    -------
    validation: dict
        Dictionary object containing accuracy and log loss metrics for each model.
    """
    # Data Imports
    team_data = pd.read_csv(Path.cwd().parent.joinpath('data', 'interim', 'team_data.csv'))
    directory = Path.cwd().parent.joinpath('reports', 'figures')

    # Validation Individual Core Models
    te_acc, te_lls = validate_team_elo(team_data, directory, graph)
    pe_acc, pe_lls = validate_player_elo(team_data, directory, graph)
    ts_acc, ts_lls = validate_trueskill(team_data, directory, graph)
    ed_acc, ed_lls = validate_egpm_dominance(team_data, directory, graph)
    se_acc, se_lls = validate_side_ema(team_data, directory, graph)

    # Validate Ensemble Prediction
    es_acc, es_lls = validate_ensemble_accuracy(team_data, te_acc, pe_acc, ts_acc, ed_acc, se_acc, directory, graph)

    # Generate Validation Variable
    validation = {"team_accuracy": te_acc, "team_logloss": te_lls,
                  "player_accuracy": pe_acc, "player_logloss": pe_lls,
                  "trueskill_accuracy": ts_acc, "trueskill_logloss": ts_lls,
                  "egpm_dom_accuracy": ed_acc, "egpm_dom_logloss": ed_lls,
                  "side_ema_accuracy": se_acc, "side_ema_logloss": se_lls,
                  "ensemble_accuracy": es_acc, "ensemble_logloss": es_lls}

    return validation


# Main
def main():
    # Data Imports
    team_data = pd.read_csv(Path.cwd().parent.joinpath('data', 'interim', 'team_data.csv'))
    directory = Path.cwd().parent.joinpath('reports', 'figures')

    # Validation Individual Core Models
    te_acc, te_lls = validate_team_elo(team_data, directory, True)
    pe_acc, pe_lls = validate_player_elo(team_data, directory, True)
    ts_acc, ts_lls = validate_trueskill(team_data, directory, True)
    ed_acc, ed_lls = validate_egpm_dominance(team_data, directory, True)
    se_acc, se_lls = validate_side_ema(team_data, directory, True)

    # Validate Ensemble Prediction
    validate_ensemble_accuracy(team_data, te_acc, pe_acc, ts_acc, ed_acc, se_acc, directory, True)


if __name__ in ('__main__', '__builtin__', 'builtins'):
    main()
    print("Validation complete.")
