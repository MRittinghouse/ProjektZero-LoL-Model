# Housekeeping
import configurations as conf
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional


# Class Definition
@dataclass
class Team:
    name: Optional[str] = None
    side: Optional[str] = None
    top: Optional[str] = None
    jng: Optional[str] = None
    mid: Optional[str] = None
    bot: Optional[str] = None
    sup: Optional[str] = None
    warning: Optional[str] = None

    # Function Definitions
    def _get_last_roster(self, player_data: pd.DataFrame) -> list:
        """
        Compute last starting roster for team from the dataframe of player data.

        Parameters
        ----------
        player_data : Pandas DataFrame
            Pandas DataFrame containing data subset to contain only player records.

        Returns
        -------
        last_starting : list
            List of players who most recently played with that team.
            Order will ALWAYS be bot, jng, mid, sup, top (alphabetical)
        """
        player_data = player_data[player_data["teamname"] == self.name].reset_index(drop=True)
        last_played = (player_data.sort_values(["date", "teamname", "position"])
                       .drop_duplicates(subset=["teamname", "position"], keep="last", ignore_index=True)
                       .reset_index())
        last_starting = list(last_played.playername.unique())

        return last_starting

    def __post_init__(self):
        # Data Import
        player_data = pd.read_csv(f"{conf.workingdir}\\ModelData\\flattened_players.csv")
        team_data = pd.read_csv(f"{conf.workingdir}\\ModelData\\flattened_teams.csv")

        if self.name:
            team_data = team_data[team_data["teamname"] == self.name].reset_index(drop=True)
            roster = self._get_last_roster(player_data)
            self.team_elo = team_data.team_elo.mean()
            self.team_trueskill_mu = team_data.trueskill_sum_mu.mean()
            self.team_trueskill_sigma = team_data.trueskill_sum_sigma.mean()

        if not self.bot:
            self.bot = roster[0]
        if not self.jng:
            self.jng = roster[1]
        if not self.mid:
            self.mid = roster[2]
        if not self.sup:
            self.sup = roster[3]
        if not self.top:
            self.top = roster[4]

        players = [self.top, self.jng, self.mid, self.bot, self.sup]

        data = player_data[player_data["playername"].isin(players)]
        if len(data) < 5:
            diff = np.setdiff1d(players, list(data.playername.unique()))
            for d in diff:
                substitute = {'date': '1/1/2022 23:59', 'teamname': 'Null', 'position': 'Null',
                              'playername': d, 'player_elo': 1100, 'trueskill_mu': 21, 'trueskill_sigma': 8,
                              'egpm_dominance_ema': 198, 'blue_side_ema': 0.4, 'red_side_ema': 0.4}
                data = data.append(substitute, ignore_index=True)
            self.warning = f"WARNING: {str(diff)} not found in database. Substitute values were used."

        self.player_elo = data.player_elo.mean()
        self.player_trueskill_mu = data.trueskill_mu.sum()
        self.player_trueskill_sigma = data.trueskill_sigma.sum()
        self.egpm_dominance = data.egpm_dominance_ema.sum()
        self.side_win_rate = data.blue_side_ema.mean() if self.side.lower() == "blue" else data.red_side_ema.mean()
