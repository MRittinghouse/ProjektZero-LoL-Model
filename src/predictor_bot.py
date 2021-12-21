# -*- coding: utf-8 -*-
"""
LoL Predictor Discord Bot
Created by ProjektZero
07/21/2021

This bot is a wrapper for my League of Legends esports prediction model.
It is intended to allow users to call down predictions. 
"""

# Housekeeping
import ast
import discord
from discord.ext import commands
from dotenv import load_dotenv
from os import getenv
import src.match_predictor as mp
import nest_asyncio

nest_asyncio.apply()

# Variable Definitions
load_dotenv()
token = getenv("DISCORD_TOKEN")

client = discord.Client()
helper = """Compute win probabilities based on Elo, TrueSkill, and Earned Gold Dominance"""

bot = commands.Bot(command_prefix='!')


# Party Time
@bot.command(name='predict')
async def predict(ctx, blue1, blue2, blue3, blue4, blue5, red1, red2, red3, red4, red5):
    prelim = "```Calculating...```"
    message = await ctx.send(content=prelim)

    try:
        output = mp.main(blue1, blue2, blue3, blue4, blue5, red1, red2, red3, red4, red5)
    except Exception as e:
        output = f"""```Something went wrong, sorry about that.
If this is still breaking, ping ProjektZero for support.
    Error:
    {e}```
    """
    await message.edit(content=output)


bot.run(token)
