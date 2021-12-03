# -*- coding: utf-8 -*-
"""
Configurations.

These are the core variables that you need to execute the main script.
"""
# Variable Definitions
r"""
The working directory, or workingdir, represents the folder name  / filepath
where you want data to be saved. This will include all outputs, including the
raw data, graphs, model validation metrics, and spreadsheets.

An example would be: 'C:\\Users\\YourMachineName\Documents\\OraclesElixir'
Note: Please don't end the path with \\, or things will break.
"""
workingdir = "C:\\Users\\matth\\Documents\\OraclesElixir"


"""
These next two variables are used to automatically pull down upcoming matches.
There are two APIs currently being utilized, one is private and the other is
used if you do not have access to the private API (you probably don't)

The "days" argument is for the Private API, and pulls down a number of
upcoming days

The "games" argument is for the public API, and pulls down a set number of
upcoming games. You can define both, there is no harm in that. Only one will
actually be used.

If you are aware of any free, public APIs that can be used to pull down the
schedule for upcoming matches, please let me know as it would help a lot!
"""
# Sets the time window. e.g: "all matches in the next 3 days"
# Used only in upcoming_schedule)
days = 2

"""
The team replacements variable is used to replace values in the data in the
event of a team name change within the dataset.
This is particularly useful when looking at multiple splits of data.
It's important to note that the default setting for the model is to use two
years of data, so you'll want to have corrections here for all major team
name changes in the last two years of gameplay.

The format is intended to be {'oldname1': 'newname1', 'oldname2': 'newname2'}
Be sure that the spelling, spacing, and capitalization match Oracle's Elixir.
"""
team_replacements = {"Dignitas": "Dignitas QNTMPAY", "eStar": "Ultra Prime"}


