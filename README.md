# ProjektZero-LoL-Model
An attempt at modeling win rates by player and by team for competitive League of Legends. This model is heavily debted to the work of Tim Sevenhuysen at Oracle's Elixir. 

Please visit and support Oracle's Elixir at www.oracleselixir.com

Currently, the intended audience of this code is for developers and analysts with an academic interest in the professional League of Legends esport. 

There are more people than I can hope to name here who have helped in some way or another in helping to building out functionality, reviewing code, helping me to answer questions, or otherwise providing guidance over the course of the years that I have worked on this code, from the original version that was written in R, to the refactor in Python, and the process of maturing the code and the model. 

I'd particularly like to thank BuckeyeSundae, TZero, Addie Thompson, and many of the folks in the Oracle's Elixir Data Science community. 

## Background:

I have been a die-hard fan of League of Legends as both a game and an esport since 2010. This codebase represents the product of years of passionate work with the intention of modeling League of Legends, dating back to 2014 when I was in graduate school and learning R as my first programming language. 

There was a period in my life where my sole source of income was derived by betting on the product of a much earlier version of this model. I strongly want to emphasize that this model is intended purely for academic purposes, and this script comes with no guarantee or expectation of performance, and any use of it for betting/wagers is done entirely at the risk of the end user. Nothing in this code or its outputs constitutes financial advice. 

I am now a full time data scientist, and want to open source this model so that others have the opportunity to engage with this code and help move forward my dream of fully taking advantage of abundance of data that esports offer in comparison to traditional sports. I truly believe that in time, analysts and developers will be more fully able to capitalize on the digital nature of esports to vastly enhance performance metrics and predictive capabilities, and that truly meaningful insights about the nature of human variability and performance in esports can be sussed out. 

## Goals:

My intent is to help improve the predictive capabilities and insights generated from this data set. I encourage anyone with an interest to get involved, submit comments, clone and work with this code. There is wisdom in crowds, and I hope that in time this project can represent a well-maintained analytics platform. 

## Structure:

This project is divided into three main modules - the first is named "oracleselixir", the second is named "lolmodeling", and the third is "dfsoptimizer". 

The "oracleselixir" module represents a series of functions designed to pull down data from the Oracle's Elixir site. That code will subset, format and clean that data, and is capable of handling file management around storing the data locally so as to minimize the volume of pulls against the Oracle's Elixir site. It is intended that the code in that module be a respectful steward of data, but also a consistent core of information for use in additional analytics. 

The "lolmodeling" module represents an ensemble of tools and modeling functions built around predicting game or match win probabilities (sometimes referred to as "moneyline" bets) and computing player and team performance statistics. In the future, this toolkit may be expanded to handle a vast number of additional metrics, including metrics around kills, objectives, prop bets, etc. 

The "dfsoptimizer" module contains a series of tools designed to take the output of lolmodeling metrics and help build out Daily Fantasy Sport (dfs) rosters. There are a wide variety of DFS sites, some free and some paid, with many variations of rules. Currently, this code supports two variants, one for "EsportsOne" (free) and another for "DraftKings" (paid). It is again strongly emphasized that this script comes with no expectation or guarantee of performance, and that the user is fully responsible for however they choose to use this script. 

Lastly, the "ProjektZero_LeagueModel.py" script represents the main execution body of the code, and its settings and specifications can be adjusted in the "configuration" file, which has detailed descriptions of each of the individual settings. 

## Models:

This project represents an ensemble model - that is, a model composed of multiple models. Initially, I tried a number of these models individually, hoping that some might outcompete the others and I'd find some "truly predictive" main model. But the more I worked, the more apparent it became to me that each model had individual strengths, weaknesses, and biases. 

For example, some models were highly sensitive to player substitutions, roster swaps, and role swaps. Other models were more representative of subtle factors like coach and supporting staff changes, player synergies, and other less tangible effects. I found myself going back and forth on whether or not to measure performance at a team level or at a player level frequently. 

Furthermore, I wanted to stay up to date on methods that were considered effective by the consensus of minds in the field. This lead me to investigate elo models, and eventually look at more proprietary ideas like TrueSkill. 

Thus, the current model is comprised of four major models:

### Team-Based Elo (for the current year)

This model looks at a shorter period of data. The K value was set by a lot of iterative trial and error, testing until I found the best fit. 

Current Tested Accuracy: 60.25%, Log Loss: 65.15% 

### Player Based Elo (for the last two years)

This model looks at the performance of each individual player. This model is more resistant to players changing teams, or moving back and forth between academy and main leagues. However, this model also has the issue of incorporating the effects of the other 4 players on the team into the player's elo. 

Current Tested Accuracy: 61.32%, Log Loss: 65.45%

### TrueSkill (for the last two years)

TrueSkill is calculated on a player-basis for the past two years. TrueSkill is much better oriented to monitor player-level effects and skill, and also uses mu and sigma values to capture a player's mean and variance in their performance. 

Current Tested Accuracy: 63.03%, Log Loss: 68.24%

### TrueSkill-Normalized EGPM Dominance (for the last two years)

This model is one that I developed myself, and is slightly more difficult to explain. But the general idea is that gold is the most individually significant stat to monitor in League of Legends as an esport. This model looks at a team's gold lead over their opponent, and uses that as a proxy to measure the "strength of the win". EGPM stands for "Earned Gold Per Minute" and is a stat measured in the Oracle's Elixir data. The intent is to monitor a team's Earned Gold Per Minute over their opponent, relative to each team's average EGPM value. The way this is set up is that if the 1st place team in the league loses to the 10th place team in the league, where the 10th place team has a much higher EGPM statistic, the 1st place team is penalized significantly more than if they lost in a close game to the 2nd place team in the league. This is similar to elo, but the EGPM statistic is used to quantify the "strength of the win". This has been the best performing model, but again may struggle with player substitutions. 

Current Tested Accuracy: 70.22%, Log Loss: 64.74%

### How It Gets Ensembled

Each of the four models' predictions is output in the document, so you can see the individual performance. This output also comes with an average that's calculated by weighting each model's accuracy into an average of all four model probability scores. A standard deviation value is also computed, so you get the ensemble model's weighted average, and standard deviation, alongside each of the four individual model probability scores as well. 
 