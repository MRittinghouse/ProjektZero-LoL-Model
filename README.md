# ProjektZero-LoL-Model
An attempt at modeling win rates by player and by team for competitive League of Legends. This model is heavily debted to the work of Tim Sevenhuysen at Oracle's Elixir. 

Please visit and support Oracle's Elixir at www.oracleselixir.com

Currently, the intended audience of this code is for developers and analysts with an academic interest in the professional League of Legends esport. 

There are more people than I can hope to name here who have helped in some way or another in reviewing code, helping me to answer questions, or otherwise providing guidance over the course of the years that I have worked on this code, from the original version that was written in R, to the refactor in Python, and the process of maturing the code and the model. 

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