#!/usr/bin/env python
# coding: utf-8

#For reading and manipulating our data
import pandas as pd

#Our visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

#Gives us access to some common math functions 
import math

#For the random component of our simulation
import random as rnd

#Neatly presents our simulation results
from tabulate import tabulate

df = pd.read_csv("UFC Matches April 2023.csv")

# Data Cleaning

# delete first row of every fighter
nulls = df[["Matchup", "Outcome", 'Method', 'Round', 'Time']].isnull().all(axis=1) 
df = df.loc[~nulls, :]

#find matches that are upcoming and delete them
df = df[df["Outcome"] != 'NEXT']

# delete "weight text" in weight column
df['Weight'] = df['Weight'].str.replace('WEIGHT: ', '')

# keep only opponents name in matchup column
def remove_name(orig_matchup):
    last_index = orig_matchup.rfind('\n')
    new_string = orig_matchup[last_index +1:]
    return new_string.strip()

df['Matchup'] = df['Matchup'].apply(lambda x: remove_name(x))

#discovered there is a duplicate fighter name 
df.loc[[19227, 19228, 19229, 19230, 19231], 'Full Name'] = "Bruno Blindado Silva"

unique_methods = df['Method'].unique()
unique_outcomes = df['Outcome'].unique()

#types of outcomes we don't want in our dataset
removed_outcomes = ['NC', 'DRAW']
removed_methods = ['Overturned', 'Other', 'CNC', 'DQ', 'Overturned ', 'CNC ']

filtered_df1 = df.loc[~df['Method'].isin(removed_methods)]
filtered_df2 = filtered_df1.loc[~df['Outcome'].isin(removed_outcomes)]
filtered_df2['Method'] = filtered_df2['Method'].apply(lambda x: x.strip())

#mapping weights to weight class label and assigning unknown to empty weight entries

filtered_df2.loc[df['Weight'] == '--', 'Weight Class'] = 'Unknown'

weight_mapping = {
    'Strawweight': range(0, 116),
    'Flyweight': range(116, 126),
    'Bantamweight': range(126, 136),
    'Featherweight': range(136, 146),
    'Lightweight': range(146, 156),
    'Welterweight': range(156, 171),
    'Middleweight': range(171, 186),
    'Light Heavyweight': range(186, 206),
    'Heavyweight': range(206, 266)
}

mask = filtered_df2['Weight'] != '--'
filtered_df2.loc[mask, 'Weight'] = filtered_df2.loc[mask, 'Weight'].str.replace('lbs.', '').astype(int)
filtered_df2['Weight Class'] = filtered_df2['Weight'].apply(lambda x: next((k for k, v in weight_mapping.items() if x in v), 'Unknown'))

#group specific outcomes to the three notable methods decision, ko/tko and submission

decisions = ['U-DEC', 'S-DEC', 'M-DEC', 'Decision']
kos = ['KO/TKO']
submissions = ['SUB']

filtered_df2['Method'] = filtered_df2['Method'].replace(decisions, 'Decision')
filtered_df2['Method'] = filtered_df2['Method'].replace(submissions, 'Submission')

df = filtered_df2

# Exploratory Data Analysis (EDA)

# What is the most common method of winning?

decisions = ['U-DEC', 'S-DEC', 'M-DEC', 'Decision']
kos = ['KO/TKO']
submissions = ['Submission']

# create a dictionary to store the frequency of each category
categories = {'Decision': len([x for x in df["Method"] if x in decisions]),
              'KO/TKO': len([x for x in df["Method"] if x in kos]),
              'Submission': len([x for x in df["Method"] if x in submissions])}

#our bar plot with percentage labels
ax = sns.barplot(x=list(categories.keys()), y=list(categories.values()))
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(df)), (p.get_x()+0.25, p.get_y()+p.get_height()/2))

plt.title("Frequency of Win Methods in the UFC")
plt.xlabel("Method")
plt.ylabel("Frequency")
plt.show()

# For fighters that win through ko/submission what round does it typically occur in?

# Submission

# filter for only matches that end in submission
only_submissions = df[df["Method"] == "Submission"]

# group by round and count submission frequency
round_freq = only_submissions.groupby("Round")["Round"].count()

# our bar plot with percentages
ax = sns.barplot(x=round_freq.index, y=round_freq.values)
total = round_freq.sum()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+1, '{:.4f}%'.format(height/total*100), ha="center")

plt.title("Submission Frequency by Round")
plt.xlabel("Round")
plt.ylabel("Frequency")
plt.show()

# KO

# filter for only matches that end in ko/tko
only_kos = df[df["Method"] == "KO/TKO"]

# group by round and count frequency
round_freq = only_kos.groupby("Round")["Round"].count()

# our bar plot with percentages
ax = sns.barplot(x=round_freq.index, y=round_freq.values)
total = round_freq.sum()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+1, '{:.4f}%'.format(height/total*100), ha="center")

plt.title("KO/TKO Frequency by Round")
plt.xlabel("Round")
plt.ylabel("Frequency")
plt.show()

# Is there a discrepancy in the way fighters win across different weight classes?

data = {'Weight Class': df["Weight Class"],'Method': df["Method"]}
weight_method_df = pd.DataFrame(data)

# a crosstab to count the frequency of win method by weight class
ct = pd.crosstab(weight_method_df['Weight Class'], weight_method_df['Method'])

# order for x-axis display
order = ['Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']
ct = ct.reindex(order)

# our stacked bar chart
ax = ct.plot(kind='bar', stacked=True)

# total count for each weight class
totals = ct.sum(axis=1)

# set the chart title and axis labels
ax.set_title('Win Method by Weight Class')
ax.set_xlabel('Weight Class')
ax.set_ylabel('Frequency')
ax.grid(False)
plt.show()

# The tabulated version of the plot above

data = {'Weight Class': df["Weight Class"], 'Method': df["Method"]}
weight_method_df = pd.DataFrame(data)

ct.index.name = None
ct.columns.name = None

# our table with percentages formatted as strings
table = ct.applymap(lambda x: '{:.2f}%'.format(100*x))
print(table)

#Get the details of each figther's match history
def transform_dataset(df):
    data = {
    'Total Fights': df.groupby(['Full Name'])['Matchup'].count(),
    'Total Wins': df.loc[df['Outcome'] == 'WIN'].groupby(['Full Name'])['Outcome'].count(),
    'Total Losses': df.loc[df['Outcome'] == 'LOSS'].groupby(['Full Name'])['Outcome'].count(),
    'Wins By Decision': df.loc[(df['Outcome'] == 'WIN') & (df['Method'].isin(decisions))].groupby(['Full Name'])['Outcome'].count(),
    'Wins By KO': df.loc[(df['Outcome'] == 'WIN') & (df['Method'].isin(kos))].groupby(['Full Name'])['Outcome'].count(),
    'Wins By Submission': df.loc[(df['Outcome'] == 'WIN') & (df['Method'].isin(submissions))].groupby(['Full Name'])['Outcome'].count(),
    'Loss By Decision': df.loc[(df['Outcome'] == 'LOSS') & (df['Method'].isin(decisions))].groupby(['Full Name'])['Outcome'].count(),
    'Loss By KO': df.loc[(df['Outcome'] == 'LOSS') & (df['Method'].isin(kos))].groupby(['Full Name'])['Outcome'].count(),
    'Loss By Submission': df.loc[(df['Outcome'] == 'LOSS') & (df['Method'].isin(submissions))].groupby(['Full Name'])['Outcome'].count()
}
    fighter_stats = pd.DataFrame(data)
    fighter_stats.fillna(0, inplace=True)
    return fighter_stats

df = transform_dataset(df)
df.head()

# The beginning of our Monte Carlo Simulation

# sample proportion p calculation
def calculate_p(statistic, total_fights):
    return statistic/total_fights

# sample sd calculation
def calculate_sd(statistic, total_fights):
    p = calculate_p(statistic, total_fights)
    return math.sqrt((p*(1-p))/total_fights)

def get_fighter_parameters(fighter1, fighter2):
    data = {
        
        "Decision Wins Prop" : [calculate_p(df.loc[fighter1, "Wins By Decision"], df.loc[fighter1, "Total Fights"]),
                                   calculate_p(df.loc[fighter2, "Wins By Decision"], df.loc[fighter2, "Total Fights"])], 
            
        "Decision Wins SD" : [calculate_sd(df.loc[fighter1, "Wins By Decision"], df.loc[fighter1, "Total Fights"]),
                                   calculate_sd(df.loc[fighter2, "Wins By Decision"], df.loc[fighter2, "Total Fights"])],
        
        "KO Wins Prop" : [calculate_p(df.loc[fighter1, "Wins By KO"], df.loc[fighter1, "Total Fights"]),
                                   calculate_p(df.loc[fighter2, "Wins By KO"], df.loc[fighter2, "Total Fights"])], 
            
        'KO Wins SD' : [calculate_sd(df.loc[fighter1, "Wins By KO"], df.loc[fighter1, "Total Fights"]),
                                   calculate_sd(df.loc[fighter2, "Wins By KO"], df.loc[fighter2, "Total Fights"])], 
           
        'Sub Wins Prop' : [calculate_p(df.loc[fighter1, "Wins By Submission"], df.loc[fighter1, "Total Fights"]),
                                   calculate_p(df.loc[fighter2, "Wins By Submission"], df.loc[fighter2, "Total Fights"])],
           
        'Sub Wins SD' : [calculate_sd(df.loc[fighter1, "Wins By Submission"], df.loc[fighter1, "Total Fights"]),
                                   calculate_sd(df.loc[fighter2, "Wins By Submission"], df.loc[fighter2, "Total Fights"])], 
            
        'Decision Loss Prop' : [calculate_p(df.loc[fighter1, "Loss By Decision"], df.loc[fighter1, "Total Fights"]),
                                   calculate_p(df.loc[fighter2, "Loss By Decision"], df.loc[fighter2, "Total Fights"])], 
            
        'Decision Loss SD' : [calculate_sd(df.loc[fighter1, "Loss By Decision"], df.loc[fighter1, "Total Fights"]),
                                   calculate_sd(df.loc[fighter2, "Loss By Decision"], df.loc[fighter2, "Total Fights"])], 
            
        'KO Loss Prop' : [calculate_p(df.loc[fighter1, "Loss By KO"], df.loc[fighter1, "Total Fights"]),
                                   calculate_p(df.loc[fighter2, "Loss By KO"], df.loc[fighter2, "Total Fights"])], 
            
        'KO Loss SD' : [calculate_sd(df.loc[fighter1, "Loss By KO"], df.loc[fighter1, "Total Fights"]),
                                   calculate_sd(df.loc[fighter2, "Loss By KO"], df.loc[fighter2, "Total Fights"])], 
           
        'Sub Loss Prop' : [calculate_p(df.loc[fighter1, "Loss By Submission"], df.loc[fighter1, "Total Fights"]),
                                   calculate_p(df.loc[fighter2, "Loss By Submission"], df.loc[fighter2, "Total Fights"])], 
            
        'Sub Loss SD' : [calculate_sd(df.loc[fighter1, "Loss By Submission"], df.loc[fighter1, "Total Fights"]),
                                   calculate_sd(df.loc[fighter2, "Loss By Submission"], df.loc[fighter2, "Total Fights"])],
        'Number of Matches' : [df.loc[fighter1, "Total Fights"], df.loc[fighter2, "Total Fights"]]
           
           }
    
    fighter_parameter_df = pd.DataFrame(data=data, index = [fighter1, fighter2])
    return fighter_parameter_df

#binomial prop

#Simulates a single fight and returns the highest scored outcome whether it be fighter 1 wins by decision, fighter 2 wins by submission etc..
def matchSim(matchup_df): 
    results = []

    # Averages the random sample of a fighters's dec/ko/sub win score with a random sample of an opponent's dec/ko/sub loss score
    # Randomly samples from the two gaussian distributions to produce a probabilistic outcome
    fighter1_dec_score = (rnd.gauss(matchup_df.iloc[0]['Decision Wins Prop'],matchup_df.iloc[0]['Decision Wins SD'])+ 
                              rnd.gauss(matchup_df.iloc[1]['Decision Loss Prop'],matchup_df.iloc[1]['Decision Loss SD']))/2
    fighter1_ko_score = (rnd.gauss(matchup_df.iloc[0]['KO Wins Prop'],matchup_df.iloc[0]['KO Wins SD'])+ 
                              rnd.gauss(matchup_df.iloc[1]['KO Loss Prop'],matchup_df.iloc[1]['KO Loss SD']))/2
    fighter1_sub_score = (rnd.gauss(matchup_df.iloc[0]['Sub Wins Prop'],matchup_df.iloc[0]['Sub Wins SD'])+ 
                              rnd.gauss(matchup_df.iloc[1]['Sub Loss Prop'],matchup_df.iloc[1]['Sub Loss SD']))/2
    fighter2_dec_score = (rnd.gauss(matchup_df.iloc[1]['Decision Wins Prop'],matchup_df.iloc[1]['Decision Wins SD'])+ 
                              rnd.gauss(matchup_df.iloc[0]['Decision Loss Prop'],matchup_df.iloc[0]['Decision Loss SD']))/2
    fighter2_ko_score = (rnd.gauss(matchup_df.iloc[1]['KO Wins Prop'],matchup_df.iloc[1]['KO Wins SD'])+ 
                              rnd.gauss(matchup_df.iloc[0]['KO Loss Prop'],matchup_df.iloc[0]['KO Loss SD']))/2
    fighter2_sub_score = (rnd.gauss(matchup_df.iloc[1]['Sub Wins Prop'],matchup_df.iloc[1]['Sub Wins SD'])+ 
                              rnd.gauss(matchup_df.iloc[0]['Sub Loss Prop'],matchup_df.iloc[0]['Sub Loss SD']))/2   
    
    results.append(fighter1_dec_score)
    results.append(fighter1_ko_score)
    results.append(fighter1_sub_score)    
    results.append(fighter2_dec_score)    
    results.append(fighter2_ko_score)    
    results.append(fighter2_sub_score)

    if max(results) == results[0]:
        return "f1_dec"
    elif max(results) == results[1]:
        return "f1_ko"
    elif max(results) == results[2]:
        return "f1_sub"
    elif max(results) == results[3]:
        return "f2_dec"
    elif max(results) == results[4]:
        return "f2_ko"
    elif max(results) == results[5]:
        return "f2_sub"
    else: return rnd.choice(["f1_dec", "f1_ko", "f1_sub", "f2_dec", "f2_ko", "f2_sub"])


def matchesSim(matchup_df, ns):
    matchesout = []
    fighter1_decwin = 0
    fighter1_kowin = 0
    fighter1_subwin = 0
    fighter2_decwin = 0
    fighter2_kowin = 0
    fighter2_subwin = 0
    tie = 0
    for i in range(ns):
        gm = matchSim(matchup_df)
        matchesout.append(gm)
        if gm == "f1_dec":
            fighter1_decwin += 1
        elif gm == "f1_ko":
            fighter1_kowin += 1
        elif gm == "f1_sub":
            fighter1_subwin += 1
        elif gm == "f2_dec":
            fighter2_decwin += 1
        elif gm == "f2_ko":
            fighter2_kowin += 1
        else:
            fighter2_subwin += 1

    results_table = [
        ["Fighter", "Win Method", "Probability"],

        [matchup_df.index[0], "Decision", str(round((fighter1_decwin / ns) * 100, 3)) + '%'],
        [matchup_df.index[0], "KO", str(round((fighter1_kowin / ns) * 100, 3)) + '%'],
        [matchup_df.index[0], "Submission", str(round((fighter1_subwin / ns) * 100, 3)) + '%'],
        [matchup_df.index[1], "Decision", str(round((fighter2_decwin / ns) * 100, 3)) + '%'],
        [matchup_df.index[1], "KO", str(round((fighter2_kowin / ns) * 100, 3)) + '%'],
        [matchup_df.index[1], "Submission", str(round((fighter2_subwin / ns) * 100, 3)) + '%']]

    print(tabulate(results_table, headers="firstrow", tablefmt="fancy_grid"))
