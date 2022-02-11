# Housekeeping
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
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
    warning: Optional[str] = ""

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
        lower_name = str(self.name).lower()
        player_data = player_data[player_data.teamname.str.lower().isin([lower_name])].reset_index(drop=True)
        last_played = (player_data.sort_values(["date", "teamname", "position"])
                       .drop_duplicates(subset=["teamname", "position"], keep="last", ignore_index=True)
                       .reset_index())
        last_starting = list(last_played.playername.unique())

        return last_starting

    def __post_init__(self):
        # Data Import
        self.team_exists = False
        if not self.side:
            self.side = "Blue"
        team_data = pd.read_csv(Path.cwd().parent.joinpath('data', 'processed', 'flattened_teams.csv'))
        lower_name = str(self.name).lower()
        team_data = team_data[team_data.teamname.str.lower().isin([lower_name])].reset_index(drop=True)
        player_data = pd.read_csv(Path.cwd().parent.joinpath('data', 'processed', 'flattened_players.csv'))

        if len(team_data.index) > 0:
            roster = self._get_last_roster(player_data)
            self.team_exists = True
            self.team_elo = team_data.team_elo.mean()
            self.team_egpm_dom = team_data.egpm_dominance.mean()
        elif lower_name in ["first 5", "second 5"]:
            pass
        else:
            self.warning += f"""\n WARNING: Team "{str(self.name)}" not found in database. No team data was used."""
            print(self.warning)

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
        players = [s.lower() for s in players]

        data = player_data[player_data.playername.str.lower().isin(players)]
        data = (data.sort_values(['playername', 'date'])
                .groupby(['playername'])
                .tail(1)
                .reset_index(drop=True))
        if len(data) > 5:
            most_common_league = data.league.mode().iloc[0]
            data = data[data['league'] == most_common_league].reset_index(drop=True)
        if len(data) < 5:
            df_players = list(data.playername.str.lower().unique())
            diff = np.setdiff1d(players, df_players)
            for d in diff:
                substitute = {'date': '1/1/2022 23:59', 'teamname': 'Null', 'position': 'Null',
                              'playername': d, 'player_elo': 1100, 'trueskill_mu': 21, 'trueskill_sigma': 8,
                              'egpm_dominance': 198, 'blue_side_ema_after': 0.4, 'red_side_ema_after': 0.4}
                data = data.append(substitute, ignore_index=True)
            self.warning += f"\n WARNING: {str(diff)} not found in database. Substitute values were used."
        elif len(data) > 5:
            raise ValueError(f'Team cannot have more than 5 player values. \n \n {data}')

        self.player_elo = data.player_elo.mean()
        self.player_trueskill_mu = data.trueskill_mu.sum()
        self.player_trueskill_sigma = data.trueskill_sigma.to_list()
        self.player_egpm_dom = data.egpm_dominance.sum()
        self.side_win_rate = data.blue_side_ema_after.mean() if self.side.lower() == "blue" \
            else data.red_side_ema_after.mean()


if __name__ in ('__main__', '__builtin__', 'builtins'):
    print(Team("Oh My God"))
