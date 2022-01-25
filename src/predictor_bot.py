# -*- coding: utf-8 -*-
"""
This bot is a wrapper for my League of Legends esports prediction model.
It is intended to allow users to call down predictions. 
"""

# Housekeeping
import discord
from discord.ext import commands
from dotenv import load_dotenv
from os import getenv
import pandas as pd
from pathlib import Path
import src.match_predictor as mp
import src.model_validator as mv
from src.team import Team
import nest_asyncio

nest_asyncio.apply()
pd.set_option("display.max_rows", None, "display.max_columns", None)

# Variable Definitions
load_dotenv()
token = getenv("DISCORD_TOKEN")

client = discord.Client()
helper = """This is ProjektZero's bot. Ask him about it. Eventually I'll put a real readme here."""

bot = commands.Bot(command_prefix='!')


# Party Time
@bot.command(name='schedule')
async def schedule(ctx, league):
    try:
        output = pd.read_csv(Path.cwd().parent.joinpath('data', 'processed', 'schedule.csv'))
        output = output[output["league"] == league].drop(['league'], axis=1).reset_index(drop=True)
        output = f"Upcoming {league} Games (Next 10 Games Within 5 Days): \n \n" \
                 f"`{output.head(10).to_markdown()}` \n \n" \
                 "`NOTE: Win percentages use the last fielded roster! Try !predict if you need substitutions.`"
    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"

    await ctx.send(content=output)


@bot.command(name='profile')
async def profile(ctx, entity):
    try:
        players = pd.read_csv(Path.cwd().parent.joinpath('data', 'processed', 'flattened_players.csv'))
        teams = pd.read_csv(Path.cwd().parent.joinpath('data', 'processed', 'flattened_teams.csv'))

        players_list = [p.lower() if isinstance(p, str) else p for p in players.playername.unique()]
        teams_list = [t.lower() if isinstance(t, str) else t for t in teams.teamname.unique()]
        lower_entity = str(entity).lower()

        if lower_entity in players_list:
            data = players[players.playername.str.lower() == lower_entity].to_dict(orient="records")[-1]
            output = f"{entity} Profile: \n \n" \
                     f"`Position: {data['position']} \n" \
                     f"Elo: {data['player_elo']:.2f} \n" \
                     f"TrueSkill Mu: {data['trueskill_mu']:.2f} \n" \
                     f"EGPM Dominance: {data['egpm_dominance']:.2f} \n" \
                     f"K/D/A Ratio: {data['kda']:.2f} \n" \
                     f"Gold Diff At 15: {data['golddiffat15']:.2f} \n" \
                     f"CS Diff At 15: {data['csdiffat15']:.2f} \n" \
                     f"DK Points: {data['dkpoints']:.2f}`"
        elif lower_entity in teams_list:
            data = teams[teams.teamname.str.lower() == lower_entity].to_dict(orient="records")[-1]
            output = f"{entity} Profile: \n \n" \
                     f"`Elo: {data['team_elo']:.2f} \n" \
                     f"TrueSkill Mu: {data['trueskill_sum_mu']:.2f} \n" \
                     f"EGPM Dominance: {data['egpm_dominance']:.2f} \n" \
                     f"K/D/A Ratio: {data['kda']:.2f} \n" \
                     f"Gold Diff At 15: {data['golddiffat15']:.2f} \n" \
                     f"CS Diff At 15: {data['csdiffat15']:.2f} \n" \
                     f"DK Points: {data['dkpoints']:.2f}`"
        else:
            output = f"Data for {entity} not found in database. \n" \
                     "Make sure that your spelling perfectly matches the Oracle's Elixir data."
    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"

    await ctx.send(content=output)


@bot.command(name='predict')
async def predict(ctx, blue_team, blue1, blue2, blue3, blue4, blue5, red_team, red1, red2, red3, red4, red5):
    prelim = "```Calculating...```"
    message = await ctx.send(content=prelim)

    try:
        output = mp.predict(blue_team, blue1, blue2, blue3, blue4, blue5,
                            red_team, red1, red2, red3, red4, red5, False)
    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"
    await message.edit(content=output)


@bot.command(name='predict_verbose')
async def predict_verbose(ctx, blue_team, blue1, blue2, blue3, blue4, blue5, red_team, red1, red2, red3, red4, red5):
    prelim = "```Calculating...```"
    message = await ctx.send(content=prelim)

    try:
        output = mp.predict(blue_team, blue1, blue2, blue3, blue4, blue5,
                            red_team, red1, red2, red3, red4, red5, True)
    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"
    await message.edit(content=output)


@bot.command(name='mock_draft')
async def mock_draft(ctx, blue1, blue2, blue3, blue4, blue5,
                     red1, red2, red3, red4, red5):
    prelim = "```Calculating...```"
    message = await ctx.send(content=prelim)

    try:
        output = mp.mock_draft(blue1, blue2, blue3, blue4, blue5, red1, red2, red3, red4, red5)
    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"
    await message.edit(content=output)


@bot.command(name='roster')
async def roster(ctx, team):
    prelim = "```Calculating...```"
    message = await ctx.send(content=prelim)

    try:
        team_data = Team(name=team)
        players = [team_data.top, team_data.jng, team_data.mid, team_data.bot, team_data.sup]
        output = f"```{team} Last Fielded Roster: \n \n" \
                 f"{players} \n" \
                 "NOTE: This model does NOT track substitutions/roster swaps for upcoming games!```"
    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"
    await message.edit(content=output)


@bot.command(name='best_of')
async def best_of(ctx, count, team1, team1_odds, team2):
    prelim = "```Calculating...```"
    message = await ctx.send(content=prelim)

    try:
        int_count = int(count)
        odds = float(team1_odds)
        opp_odds = 1 - float(team1_odds)

        if int_count == 3:
            output = mp.best_of_three(team1, odds, team2, opp_odds)
        elif int_count == 5:
            output = mp.best_of_five(team1, odds, team2, opp_odds)
        else:
            raise ValueError("Series count must be 3 or 5.")

    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"
    await message.edit(content=output)


@bot.command(name='validation')
async def validation(ctx, method, graph):
    try:
        vld = mv.generate_validation_metrics(graph=False)
        true_vals = ["Yes", "yes", "True", "true", "1", "Graph", "graph"]

        if method.lower() == "team elo":
            acc = vld['team_accuracy']
            lls = vld['team_logloss']
            brier = vld['team_brier']
            metrics = f"`{method} Accuracy: {acc:.5f}, Log Loss: {lls:.5f}, Brier Score: {brier:.5f}`"
            if str(graph) in true_vals:
                with open(Path.cwd().parent.joinpath('reports', 'figures', 'TeamElo_Validation.png'), 'rb') as f:
                    image = discord.File(f)

        elif method.lower() == "player elo":
            acc = vld['player_accuracy']
            lls = vld['player_logloss']
            brier = vld['player_brier']
            metrics = f"`{method} Accuracy: {acc:.5f}, Log Loss: {lls:.5f}, Brier Score: {brier:.5f}`"
            if str(graph) in true_vals:
                with open(Path.cwd().parent.joinpath('reports', 'figures', 'PlayerElo_Validation.png'), 'rb') as f:
                    image = discord.File(f)

        elif method.lower() == "trueskill":
            acc = vld['trueskill_accuracy']
            lls = vld['trueskill_logloss']
            brier = vld['trueskill_brier']
            metrics = f"`{method} Accuracy: {acc:.5f}, Log Loss: {lls:.5f}, Brier Score: {brier:.5f}`"
            if str(graph) in true_vals:
                with open(Path.cwd().parent.joinpath('reports', 'figures', 'TrueSkill_Validation.png'), 'rb') as f:
                    image = discord.File(f)

        elif method.lower() == "side ema":
            acc = vld['side_ema_accuracy']
            lls = vld['side_ema_logloss']
            brier = vld['side_ema_brier']
            metrics = f"`{method} Accuracy: {acc:.5f}, Log Loss: {lls:.5f}, Brier Score: {brier:.5f}`"
            if str(graph) in true_vals:
                with open(Path.cwd().parent.joinpath('reports', 'figures', 'SideEMA_Validation.png'), 'rb') as f:
                    image = discord.File(f)

        elif method.lower() == "egpm dom":
            acc = vld['egpm_dom_accuracy']
            lls = vld['egpm_dom_logloss']
            brier = vld['egpm_dom_brier']
            metrics = f"`{method} Accuracy: {acc:.5f}, Log Loss: {lls:.5f}, Brier Score: {brier:.5f}`"
            if str(graph) in true_vals:
                with open(Path.cwd().parent.joinpath('reports', 'figures', 'EGPMDom_Validation.png'), 'rb') as f:
                    image = discord.File(f)

        elif method.lower() == "ensemble":
            acc = vld['ensemble_accuracy']
            lls = vld['ensemble_logloss']
            brier = vld['ensemble_brier']
            metrics = f"`{method} Accuracy: {acc:.5f}, Log Loss: {lls:.5f}, Brier Score: {brier:.5f}`"
            if str(graph) in true_vals:
                with open(Path.cwd().parent.joinpath('reports', 'figures', 'EnsembleModel_Validation.png'), 'rb') as f:
                    image = discord.File(f)

        else:
            raise ValueError("Method must be either team elo, player elo, trueskill, side ema, egpm dom, or ensemble.")

        if str(graph) in true_vals:
            await ctx.send(content=metrics, file=image)
        else:
            await ctx.send(content=metrics)
    except Exception as e:
        output = f"Something went wrong, sorry about that. \n" \
                 "If this is still breaking, ping ProjektZero for support. \n" \
                 "Error: \n" \
                 f"```{e}```"
        await ctx.send(content=output)


bot.run(token)
