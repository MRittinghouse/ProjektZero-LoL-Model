# -*- coding: utf-8 -*-
"""
Oracle's Elixir Match Predictor

This script is intended to query the data lake and predict upcoming games.

Please visit and support www.oracleselixir.com
Tim provides an invaluable service to the League community.
"""
# Housekeeping
import itertools
import math
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from src.team import Team
from src.model_validator import generate_validation_metrics

pd.options.display.float_format = '{:,.4f}'.format
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Load EGPM Dominance Model
filepath = Path.cwd().parent.joinpath('models', 'egpm_dom_logistic_regression.csv')
egpm_model = pickle.load(open(filepath, 'rb'))

def predict_match(blue: Team, red: Team) -> pd.DataFrame:
    def standard_prediction(blue_stat: float, red_stat: float) -> float:
        blue_win_perc = (blue_stat / (blue_stat + red_stat))
        return blue_win_perc

    def elo_prediction(blue_team_elo: float, red_team_elo: float) -> float:
        blue_win_perc = 1.0 / (1 + 10 ** ((red_team_elo - blue_team_elo) / 400))
        return blue_win_perc

    def trueskill_prediction(blue_team_mu: float, blue_team_sigma: list,
                             red_team_mu: float, red_team_sigma: list, sigma_value: float) -> float:
        delta_mu = (blue_team_mu - red_team_mu)
        sum_sig = sum(r ** 2 for r in itertools.chain(blue_team_sigma, red_team_sigma))
        denominator = math.sqrt(10 * ((sigma_value / 2) ** 2) + sum_sig)
        blue_win_perc = norm.cdf(delta_mu / denominator)
        return blue_win_perc

    def egpm_dom_prediction(blue_team_egpm_dom: float, red_team_egpm_dom: float) -> float:
        X = np.array([blue_team_egpm_dom, red_team_egpm_dom]).reshape(1, -1)
        blue_win_perc = egpm_model.predict_proba(X)[:, 1]
        return blue_win_perc

    weights = generate_validation_metrics(graph=False)

    match = pd.DataFrame({"blue": [blue.name],
                          "red": [red.name],
                          "player_elo": [elo_prediction(blue.player_elo, red.player_elo)],
                          "player_trueskill": [trueskill_prediction(blue.player_trueskill_mu,
                                                                    blue.player_trueskill_sigma,
                                                                    red.player_trueskill_mu,
                                                                    red.player_trueskill_sigma,
                                                                    sigma_value=2.75)],
                          "team_egpm_dom": [egpm_dom_prediction(blue.team_egpm_dom, red.team_egpm_dom)],
                          "player_egpm_dom": [standard_prediction(blue.player_egpm_dom, red.player_egpm_dom)],
                          "side_win": [standard_prediction(blue.side_win_rate, red.side_win_rate)]})

    if blue.team_exists and red.team_exists and blue.name.lower() != 'first 5':
        match["team_elo"] = elo_prediction(blue.team_elo, red.team_elo)
        sum_accuracy = (weights["team_accuracy"] + weights["player_accuracy"] + weights["trueskill_accuracy"] +
                        weights["egpm_dom_accuracy"] + weights["side_ema_accuracy"])

        match["blue_win_chance"] = float(((match["team_elo"] * (weights["team_accuracy"] / sum_accuracy)) +
                                          (match["player_elo"] * (weights["player_accuracy"] / sum_accuracy)) +
                                          (match["player_trueskill"] * (weights["trueskill_accuracy"] / sum_accuracy)) +
                                          (match["team_egpm_dom"] * (weights["egpm_dom_accuracy"] / sum_accuracy)) +
                                          (match["side_win"] * (weights["side_ema_accuracy"] / sum_accuracy))))

        match["deviation"] = match[["team_elo", "player_elo", "player_trueskill", "team_egpm_dom", "side_win"]].std(
            axis=1)
        match.drop(['player_egpm_dom'], axis=1, inplace=True)
    else:
        sum_accuracy = (weights["player_accuracy"] + weights["trueskill_accuracy"] +
                        weights["egpm_dom_accuracy"] + weights["side_ema_accuracy"])
        match["blue_win_chance"] = ((match["player_elo"] * (weights["player_accuracy"] / sum_accuracy)) +
                                    (match["player_trueskill"] * (weights["trueskill_accuracy"] / sum_accuracy)) +
                                    (match["player_egpm_dom"] * (weights["egpm_dom_accuracy"] / sum_accuracy)) +
                                    (match["side_win"] * (weights["side_ema_accuracy"] / sum_accuracy)))
        match["deviation"] = match[["player_elo", "player_trueskill", "player_egpm_dom", "side_win"]].std(axis=1)

    return match


def best_of_three(t1name, t1odds, t2name, t2odds):
    """
    Predict the outcome of a Best of 3 series with game odds.

    Parameters
    ----------
    t1name : str
        Team 1 Name.
    t1odds : float
        Team 1 Odds.
    t2name : str
        Team 2 Name.
    t2odds : float
        Team 2 Odds.

    Returns
    -------
    A string containing predictions and outputs for a Best of 3 series.
    Yields both the overall win likelihood, and game score probabilities.
    This is performed by taking the model output odds and applying basic
        multiple probability formulae.
    """
    # Team 1 2/0:
    t1_20 = (t1odds * t1odds)

    # Team 1 2/1
    t1_21 = (t1odds * t2odds * t1odds) + (t2odds * t1odds * t1odds)

    # Team 2 2/0
    t2_20 = (t2odds * t2odds)

    # Team 2 2/1
    t2_21 = (t2odds * t1odds * t2odds) + (t1odds * t2odds * t2odds)

    # Final Outputs
    doublecheck = t1_20 + t1_21 + t2_20 + t2_21
    assert round(doublecheck, 5) == 1.0

    output = f"```Overall Likelihood Of {t1name} To Win Series: {((t1_20 + t1_21) * 100):.2f}% \n" \
             f"Probability {t1name} wins 2/0: {(t1_20 * 100):.2f}% \n" \
             f"Probability {t1name} wins 2/1: {(t1_21 * 100):.2f}% \n \n" \
             f"Overall Likelihood Of {t2name} To Win Series: {((t2_20 + t2_21) * 100):.2f}% \n" \
             f"Probability {t2name} wins 2/0: {(t2_20 * 100):.2f}% \n" \
             f"Probability {t2name} wins 2/1: {(t2_21 * 100):.2f}%```"
    return output


def best_of_five(t1name, t1odds, t2name, t2odds):
    """
    Predict the results of a Best of 5 series using single game probability.

    Parameters
    ----------
    t1name : str
        Team 1 Name.
    t1odds : float
        Team 1 Odds.
    t2name : str
        Team 2 Name.
    t2odds : float
        Team 2 Odds.

    Returns
    -------
    A string containing predictions and outputs for a Best of 5 series.
    Yields both the overall win likelihood, and game score probabilities.
    This is performed by taking the model output odds and applying basic
        multiple probability formulae.
    """
    # Team 1 3/0:
    t1_30 = (t1odds * t1odds * t1odds)

    # Team 1 3/1
    t1_31 = ((t1odds * t1odds * t2odds * t1odds) +
             (t2odds * t1odds * t1odds * t1odds) +
             (t1odds * t2odds * t1odds * t1odds))

    # Team 1 3/2
    t1_32 = ((t2odds * t2odds * t1odds * t1odds * t1odds) +
             (t2odds * t1odds * t1odds * t2odds * t1odds) +
             (t2odds * t1odds * t2odds * t1odds * t1odds) +
             (t1odds * t2odds * t2odds * t1odds * t1odds) +
             (t1odds * t1odds * t2odds * t2odds * t1odds) +
             (t1odds * t2odds * t1odds * t2odds * t1odds))

    # Team 2 3/0
    t2_30 = (t2odds * t2odds * t2odds)

    # Team 2 3/1
    t2_31 = ((t2odds * t2odds * t1odds * t2odds) +
             (t1odds * t2odds * t2odds * t2odds) +
             (t2odds * t1odds * t2odds * t2odds))

    # Team 2 3/2
    t2_32 = ((t1odds * t1odds * t2odds * t2odds * t2odds) +
             (t1odds * t2odds * t2odds * t1odds * t2odds) +
             (t1odds * t2odds * t1odds * t2odds * t2odds) +
             (t2odds * t1odds * t1odds * t2odds * t2odds) +
             (t2odds * t2odds * t1odds * t1odds * t2odds) +
             (t2odds * t1odds * t2odds * t1odds * t2odds))

    # Final Outputs
    doublecheck = t1_30 + t1_31 + t1_32 + t2_30 + t2_31 + t2_32
    assert round(doublecheck, 5) == 1.0
    t1likelihood = ((t1_30 + t1_31 + t1_32) * 100)
    t2likelihood = ((t2_30 + t2_31 + t2_32) * 100)

    output = f"```Overall Likelihood Of {t1name} To Win Series: {t1likelihood:.2f}% \n" \
             f"Probability {t1name} wins 3/0: {(t1_30 * 100):.2f}% \n" \
             f"Probability {t1name} wins 3/1: {(t1_31 * 100):.2f}% \n" \
             f"Probability {t1name} wins 3/2: {(t1_32 * 100):.2f}% \n \n" \
             f"Overall Likelihood Of {t2name} To Win Series: {t2likelihood:.2f}% \n" \
             f"Probability {t2name} wins 3/0: {(t2_30 * 100):.2f}% \n" \
             f"Probability {t2name} wins 3/1: {(t2_31 * 100):.2f}% \n" \
             f"Probability {t2name} wins 3/2: {(t2_32 * 100):.2f}%```"
    return output


def predict(blue_team: str, blue1: str, blue2: str, blue3: str, blue4: str, blue5: str,
            red_team: str, red1: str, red2: str, red3: str, red4: str, red5: str,
            verbose: bool = False):
    output = pd.DataFrame()

    blue = Team(name=blue_team, side="Blue", top=blue1, jng=blue2, mid=blue3, bot=blue4, sup=blue5)
    red = Team(name=red_team, side="Red", top=red1, jng=red2, mid=red3, bot=red4, sup=red5)

    match = predict_match(blue, red)
    output = pd.concat([output, match], ignore_index=True)
    output["blue_win_chance"] = output["blue_win_chance"].astype('int32')

    if verbose:
        output = output[["blue", "red", "blue_win_chance", "deviation",
                         "team_elo", "player_elo", "player_trueskill", "team_egpm_dom", "side_win"]]
    else:
        output = output[["blue", "red", "blue_win_chance", "deviation"]]

    output = "```ProjektZero Model Predictions: \n \n" \
             f"{output.copy().to_markdown()}"

    if blue.warning:
        output += f"\n{blue.warning}"
    if red.warning:
        output += f"\n{red.warning}"

    output += """\n \nPlease consider supporting my obsessive coding habit at: 
    https://www.buymeacoffee.com/projektzero``` """

    # Optional Print Statements for troubleshooting
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(output)

    return output


def mock_draft(blue1: str, blue2: str, blue3: str, blue4: str, blue5: str,
               red1: str, red2: str, red3: str, red4: str, red5: str):
    output = pd.DataFrame()

    blue = Team(name="First 5", side="Blue", top=blue1, jng=blue2, mid=blue3, bot=blue4, sup=blue5)
    red = Team(name="Second 5", side="Red", top=red1, jng=red2, mid=red3, bot=red4, sup=red5)

    match = predict_match(blue, red)
    output = pd.concat([output, match], ignore_index=True)

    output = output[["blue", "red", "blue_win_chance", "deviation"]]
    output = "```ProjektZero Model Predictions:" \
             f"{output.copy()}"

    if blue.warning != '\n WARNING: Team "First 5" not found in database. No team data was used.':
        output += f"\n{blue.warning}"
    if red.warning != '\n WARNING: Team "Second 5" not found in database. No team data was used.':
        output += f"\n{red.warning}"

    output += """\n \nPlease consider supporting my obsessive coding habit at: 
    https://www.buymeacoffee.com/projektzero``` """

    # Optional Print Statements for troubleshooting
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(output)

    return output


if __name__ in ('__main__', '__builtin__', 'builtins'):
    blue_team = Team("Evil Geniuses", "Impact", "Inspired", "jojopyun", "Danny", "Vulcan")
    print(blue_team.team_egpm_dom)
    red_team = Team("Golden Guardians", "Tony Top", "Iconic", "ry0ma", "Stixxay", "Chime")
    print(red_team.team_egpm_dom)

    print(predict("Evil Geniuses", "Impact", "Inspired", "jojopyun", "Danny", "Vulcan",
                  "Golden Guardians", "Tony Top", "Iconic", "ry0ma", "Stixxay", "Chime", True))
