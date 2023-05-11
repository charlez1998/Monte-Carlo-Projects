#!/usr/bin/env python
# coding: utf-8

# In[38]:


#get data on ways fighter won/lost through dec, won/lost through sub, won/lost through ko/tko  
#tabulate outcomes in a table such as: Outcome, % chance, winner 
#FRONT END: give the user the ability to input two fighters (drop down menu)
# ---- FINISHED --- 
#extra: prevent users from picking two fighters that are from different weight classes
#--- LONG RUN --- 
# Improve model where it takes # of strikes (successful, attempted) , ground game/submissions duration, takedowns (successful, attempted)
# into consideration 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("UFC Matches April 2023.csv")

# Data Cleaning

# We have to clean our dataset then group each fighter and get their distributions to determine their likelihood of winning/losing via decision, tko/ko or submission.

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

decisions = ['U-DEC', 'S-DEC', 'M-DEC', 'Decision']
kos = ['KO/TKO']
submissions = ['SUB']

filtered_df2['Method'] = filtered_df2['Method'].replace(decisions, 'Decision')
filtered_df2['Method'] = filtered_df2['Method'].replace(submissions, 'Submission')

df = filtered_df2

# ## Exploratory Data Analysis (EDA)

# decision_df = df[df['Method'].isin(decisions)]
# decision_count = decision_df['Method'].count()

# ko_df = df[df['Method'].isin(kos)]
# ko_count = ko_df['Method'].count()

# submission_df = df[df['Method'].isin(submissions)]
# submission_count = submission_df['Method'].count()

# print(decision_count/len(df))
# print(ko_count/len(df))
# print(submission_count/len(df))

#What is the most common method of winning?

decisions = ['U-DEC', 'S-DEC', 'M-DEC', 'Decision']
kos = ['KO/TKO']
submissions = ['SUB']

# create a dictionary to store the frequency of each category
categories = {'Decision': len([x for x in df["Method"] if x in decisions]),
              'KO/TKO': len([x for x in df["Method"] if x in kos]),
              'Submission': len([x for x in df["Method"] if x in submissions])}

ax = sns.barplot(x=list(categories.keys()), y=list(categories.values()))
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(df)), (p.get_x()+0.25, p.get_y()+p.get_height()/2))

# create a bar plot using seaborn
sns.barplot(x=list(categories.keys()), y=list(categories.values()))

plt.title("Frequency of Win Methods in the UFC")
plt.xlabel("Method")
plt.ylabel("Frequency")

plt.show()

#For fighters that win through ko/submission what round does it typically occur in?

#Check for only submissions
only_submissions = df[df["Method"] == "SUB"]

# group by round and count frequency
round_freq = only_submissions.groupby("Round")["Round"].count()

# create bar plot using seaborn
ax = sns.barplot(x=round_freq.index, y=round_freq.values)
plt.title("Submission Frequency by Round")
plt.xlabel("Round")
plt.ylabel("Frequency")

total = round_freq.sum()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+1, '{:.4f}%'.format(height/total*100), ha="center")
    
plt.show()

#Check for only KO/TKO
only_kos = df[df["Method"] == "KO/TKO"]

# group by round and count frequency
round_freq = only_kos.groupby("Round")["Round"].count()

# create bar plot using seaborn
ax = sns.barplot(x=round_freq.index, y=round_freq.values)
plt.title("KO/TKO Frequency by Round")
plt.xlabel("Round")
plt.ylabel("Frequency")

total = round_freq.sum()
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+1, '{:.4f}%'.format(height/total*100), ha="center")
    
plt.show()

#is there a discrepancy in the way fighters win across different weight classes?

order = ['Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']

data = {'Weight Class': df["Weight Class"],'Method': df["Method"]}
df = pd.DataFrame(data)

# create a crosstab to count the frequency of methods by weight class
ct = pd.crosstab(df['Weight Class'], df['Method'])

ct = ct.reindex(order)

# create the stacked bar chart
ax = ct.plot(kind='bar', stacked=True)

# calculate the total count for each weight class
totals = ct.sum(axis=1)

# set the chart title and axis labels
ax.set_title('Win Method by Weight Class')
ax.set_xlabel('Weight Class')
ax.set_ylabel('Frequency')

ax.grid(False)

plt.show()

#For the tabulated version
order = ['Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight']

data = {'Weight Class': df["Weight Class"], 'Method': df["Method"]}
df = pd.DataFrame(data)

# create a crosstab to count the frequency of methods by weight class
ct = pd.crosstab(df['Weight Class'], df['Method'], normalize='index')

# re-order the rows based on the defined order of weight classes
ct = ct.reindex(order)

ct.index.name = None
ct.columns.name = None

# format the percentages as strings and create a table
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

import math
import random as rnd
import numpy.random as npr
import scipy.stats as ss

def calculate_p(statistic, total_fights):
    return statistic/total_fights

# def calculate_sd(statistic, total_fights):
#     mean = calculate_p(statistic, total_fights)
#     return math.sqrt((total_fights*mean)*(1-mean))

def calculate_sd(statistic, total_fights):
    p = calculate_p(statistic, total_fights)
    return math.sqrt((p*(1-p))/total_fights)

# def calculate_sd(statistic, total_fights):
#     p = calculate_p(statistic, total_fights)

#     if total_fights >= 15:
#         return math.sqrt((p*(1-p))/total_fights)
#     else:
#         return math.sqrt((total_fights*p)*(1-p))

def get_fighter_parameters(fighter1, fighter2):
    data = {
        
        "Decision Wins" : [df.loc[fighter1, "Wins By Decision"], df.loc[fighter2, "Wins By Decision"]],
        
        "KO Wins" : [df.loc[fighter1, "Wins By KO"], df.loc[fighter2, "Wins By KO"]],
        
        "Sub Wins" : [df.loc[fighter1, "Wins By Submission"], df.loc[fighter2, "Wins By Submission"]],
        
        "Decision Losses" : [df.loc[fighter1, "Loss By Decision"], df.loc[fighter2, "Loss By Decision"]],
        
        "KO Losses" : [df.loc[fighter1, "Loss By KO"], df.loc[fighter2, "Loss By KO"]],
        
        "Sub Losses" : [df.loc[fighter1, "Loss By Submission"], df.loc[fighter2, "Loss By Submission"]],
        
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


# In[273]:


# def gameSim(): binomial count
#     results = []
#     fighter1_dec_score = (rnd.gauss(matchup_df.iloc[0]['Decision Wins'],matchup_df.iloc[0]['Decision Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[1]['Decision Losses'],matchup_df.iloc[1]['Decision Loss SD']))/2
#     fighter1_ko_score = (rnd.gauss(matchup_df.iloc[0]['KO Wins'],matchup_df.iloc[0]['KO Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[1]['KO Losses'],matchup_df.iloc[1]['KO Loss SD']))/2
#     fighter1_sub_score = (rnd.gauss(matchup_df.iloc[0]['Sub Wins'],matchup_df.iloc[0]['Sub Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[1]['Sub Losses'],matchup_df.iloc[1]['Sub Loss SD']))/2
#     fighter2_dec_score = (rnd.gauss(matchup_df.iloc[1]['Decision Wins'],matchup_df.iloc[1]['Decision Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[0]['Decision Losses'],matchup_df.iloc[0]['Decision Loss SD']))/2
#     fighter2_ko_score = (rnd.gauss(matchup_df.iloc[1]['KO Wins'],matchup_df.iloc[1]['KO Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[0]['KO Losses'],matchup_df.iloc[0]['KO Loss SD']))/2
#     fighter2_sub_score = (rnd.gauss(matchup_df.iloc[1]['Sub Wins'],matchup_df.iloc[1]['Sub Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[0]['Sub Losses'],matchup_df.iloc[0]['Sub Loss SD']))/2   
    
#     results.append(fighter1_dec_score)
#     results.append(fighter1_ko_score)
#     results.append(fighter1_sub_score)    
#     results.append(fighter2_dec_score)    
#     results.append(fighter2_ko_score)    
#     results.append(fighter2_sub_score)
    
    
#     if max(results) == results[0]:
#         return "f1_dec"
#     elif max(results) == results[1]:
#         return "f1_ko"
#     elif max(results) == results[2]:
#         return "f1_sub"
#     elif max(results) == results[3]:
#         return "f2_dec"
#     elif max(results) == results[4]:
#         return "f2_ko"
#     elif max(results) == results[5]:
#         return "f2_sub"
#     else: return rnd.choice(["f1_dec", "f1_ko", "f1_sub", "f2_dec", "f2_ko", "f2_sub"])

# def gameSim(): binomial
#     results = []
#     fighter1_dec_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['Decision Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Decision Loss Prop']))/2
#     fighter1_ko_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['KO Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['KO Loss Prop']))/2
#     fighter1_sub_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['Sub Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Sub Loss Prop']))/2
#     fighter2_dec_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Decision Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['Decision Loss Prop']))/2
#     fighter2_ko_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['KO Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['KO Loss Prop']))/2
#     fighter2_sub_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Sub Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['Sub Loss Prop']))/2  
    
#     results.append(fighter1_dec_score)
#     results.append(fighter1_ko_score)
#     results.append(fighter1_sub_score)    
#     results.append(fighter2_dec_score)    
#     results.append(fighter2_ko_score)    
#     results.append(fighter2_sub_score)
    
    
#     if max(results) == results[0]:
#         return "f1_dec"
#     elif max(results) == results[1]:
#         return "f1_ko"
#     elif max(results) == results[2]:
#         return "f1_sub"
#     elif max(results) == results[3]:
#         return "f2_dec"
#     elif max(results) == results[4]:
#         return "f2_ko"
#     elif max(results) == results[5]:
#         return "f2_sub"
#     else: return rnd.choice(["f1_dec", "f1_ko", "f1_sub", "f2_dec", "f2_ko", "f2_sub"])

#binomial prop 
def gameSim(): 
    results = []
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

# mixed
# def gameSim(): 
#     results = []
    
#     if matchup_df.iloc[0]["Number of Matches"] >= 15 and matchup_df.iloc[1]["Number of Matches"] >= 15:
#         fighter1_dec_score = (rnd.gauss(matchup_df.iloc[0]['Decision Wins Prop'],matchup_df.iloc[0]['Decision Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[1]['Decision Loss Prop'],matchup_df.iloc[1]['Decision Loss SD']))/2
#         fighter1_ko_score = (rnd.gauss(matchup_df.iloc[0]['KO Wins Prop'],matchup_df.iloc[0]['KO Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[1]['KO Loss Prop'],matchup_df.iloc[1]['KO Loss SD']))/2
#         fighter1_sub_score = (rnd.gauss(matchup_df.iloc[0]['Sub Wins Prop'],matchup_df.iloc[0]['Sub Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[1]['Sub Loss Prop'],matchup_df.iloc[1]['Sub Loss SD']))/2
#         fighter2_dec_score = (rnd.gauss(matchup_df.iloc[1]['Decision Wins Prop'],matchup_df.iloc[1]['Decision Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[0]['Decision Loss Prop'],matchup_df.iloc[0]['Decision Loss SD']))/2
#         fighter2_ko_score = (rnd.gauss(matchup_df.iloc[1]['KO Wins Prop'],matchup_df.iloc[1]['KO Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[0]['KO Loss Prop'],matchup_df.iloc[0]['KO Loss SD']))/2
#         fighter2_sub_score = (rnd.gauss(matchup_df.iloc[1]['Sub Wins Prop'],matchup_df.iloc[1]['Sub Wins SD'])+ 
#                               rnd.gauss(matchup_df.iloc[0]['Sub Loss Prop'],matchup_df.iloc[0]['Sub Loss SD']))/2  
        
#     elif matchup_df.iloc[0]["Number of Matches"] >= 15 and matchup_df.iloc[1]["Number of Matches"] < 15:
#         fighter1_dec_score = (rnd.gauss(matchup_df.iloc[0]['Decision Wins Prop'],matchup_df.iloc[0]['Decision Wins SD'])+
#                             npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Decision Loss Prop']))/2
#         fighter1_ko_score = (rnd.gauss(matchup_df.iloc[0]['KO Wins Prop'],matchup_df.iloc[0]['KO Wins SD'])+
#                             npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['KO Loss Prop']))/2
#         fighter1_sub_score = (rnd.gauss(matchup_df.iloc[0]['Sub Wins Prop'],matchup_df.iloc[0]['Sub Wins SD'])+
#                             npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Sub Loss Prop']))/2
#         fighter2_dec_score = (rnd.gauss(matchup_df.iloc[1]['Decision Wins Prop'],matchup_df.iloc[1]['Decision Wins SD'])+ 
#                             npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['Decision Loss Prop']))/2
#         fighter2_ko_score = (rnd.gauss(matchup_df.iloc[1]['KO Wins Prop'],matchup_df.iloc[1]['KO Wins SD'])+
#                             npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['KO Loss Prop']))/2
#         fighter2_sub_score = (rnd.gauss(matchup_df.iloc[1]['Sub Wins Prop'],matchup_df.iloc[1]['Sub Wins SD'])+
#                             npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['Sub Loss Prop']))/2 
        
#     elif matchup_df.iloc[0]["Number of Matches"] < 15 and matchup_df.iloc[1]["Number of Matches"] >= 15:
#         fighter1_dec_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['Decision Wins Prop'])+ 
#                             rnd.gauss(matchup_df.iloc[1]['Decision Loss Prop'],matchup_df.iloc[1]['Decision Loss SD']))/2
#         fighter1_ko_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['KO Wins Prop'])+
#                             rnd.gauss(matchup_df.iloc[1]['KO Loss Prop'],matchup_df.iloc[1]['KO Loss SD']))/2
#         fighter1_sub_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['Sub Wins Prop'])+
#                             rnd.gauss(matchup_df.iloc[1]['Sub Loss Prop'],matchup_df.iloc[1]['Sub Loss SD']))/2
#         fighter2_dec_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Decision Wins Prop'])+
#                             rnd.gauss(matchup_df.iloc[0]['Decision Loss Prop'],matchup_df.iloc[0]['Decision Loss SD']))/2
#         fighter2_ko_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['KO Wins Prop'])+ 
#                             rnd.gauss(matchup_df.iloc[0]['KO Loss Prop'],matchup_df.iloc[0]['KO Loss SD']))/2
#         fighter2_sub_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Sub Wins Prop'])+ 
#                             rnd.gauss(matchup_df.iloc[0]['Sub Loss Prop'],matchup_df.iloc[0]['Sub Loss SD']))/2
    
#     else: 
#         fighter1_dec_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['Decision Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Decision Loss Prop']))/2
#         fighter1_ko_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['KO Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['KO Loss Prop']))/2
#         fighter1_sub_score = (npr.binomial(matchup_df.iloc[0]["Number of Matches"], matchup_df.iloc[0]['Sub Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Sub Loss Prop']))/2
#         fighter2_dec_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Decision Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['Decision Loss Prop']))/2
#         fighter2_ko_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['KO Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['KO Loss Prop']))/2
#         fighter2_sub_score = (npr.binomial(matchup_df.iloc[1]['Number of Matches'], matchup_df.iloc[1]['Sub Wins Prop'])+ 
#                               npr.binomial(matchup_df.iloc[0]['Number of Matches'], matchup_df.iloc[0]['Sub Loss Prop']))/2 
    
#     results.append(fighter1_dec_score)
#     results.append(fighter1_ko_score)
#     results.append(fighter1_sub_score)    
#     results.append(fighter2_dec_score)    
#     results.append(fighter2_ko_score)    
#     results.append(fighter2_sub_score)
    
    
#     if max(results) == results[0]:
#         return "f1_dec"
#     elif max(results) == results[1]:
#         return "f1_ko"
#     elif max(results) == results[2]:
#         return "f1_sub"
#     elif max(results) == results[3]:
#         return "f2_dec"
#     elif max(results) == results[4]:
#         return "f2_ko"
#     elif max(results) == results[5]:
#         return "f2_sub"
#     else: return rnd.choice(["f1_dec", "f1_ko", "f1_sub", "f2_dec", "f2_ko", "f2_sub"])

def gamesSim(ns):
    matchesout = []
    fighter1_decwin = 0
    fighter1_kowin = 0
    fighter1_subwin = 0
    fighter2_decwin = 0
    fighter2_kowin = 0
    fighter2_subwin = 0
    tie = 0
    for i in range(ns):
        gm = gameSim()
        matchesout.append(gm)
        if gm == "f1_dec":
            fighter1_decwin +=1 
        elif gm == "f1_ko":
            fighter1_kowin +=1 
        elif gm == "f1_sub":
            fighter1_subwin +=1 
        elif gm == "f2_dec":
            fighter2_decwin +=1 
        elif gm == "f2_ko":
            fighter2_kowin +=1 
        else:
            fighter2_subwin +=1 

    
    print(matchup_df.index[0] +' Decision Win ', round((fighter1_decwin/ns)*100,3),'%')
    print(matchup_df.index[0] +' KO Win ', round((fighter1_kowin/ns)*100,3),'%')
    print(matchup_df.index[0] +' Submission Win ', round((fighter1_subwin/ns)*100,3),'%')
    print(matchup_df.index[1] +' Decision Win ', round((fighter2_decwin/ns)*100,3),'%')
    print(matchup_df.index[1] +' KO Win ', round((fighter2_kowin/ns)*100,3),'%')
    print(matchup_df.index[1] +' Submission Win ', round((fighter2_subwin/ns)*100,3),'%')
    #print('Tie ', (tie/ns)*100, '%')
    
    #return matchesout


# Predictions on UFC Fight Night: Pavlovich vs Blaydes

# Sergei Pavlovich vs Curtis Blaydes
matchup_df = get_fighter_parameters("Sergei Pavlovich", "Curtis Blaydes")
gamesSim(10000)
#sergei ko

#perfect


# ### Brad Tavares vs Bruno Silva
matchup_df = get_fighter_parameters("Brad Tavares", "Bruno Blindado Silva")
gamesSim(10000)
#bruno silva ko

#------------------------------>= 15 swapped
#winner wrong, 2nd option correct

#------------------------------ binomial
#winner wrong, 2nd option correct

#----------------------------- binomial prop
#winner wrong, 2nd option correct

#----------------------------- binomial prop
#winner wrong, 2nd option correct


# ### Bobby Green vs Jared Gordon - No Contest

# In[280]:


#matchup_df = get_fighter_parameters("Bobby Green", "Jared Gordon")
#gamesSim(10000)


# ### Iasmin Lucindo vs Brogan Walker
matchup_df = get_fighter_parameters("Iasmin Lucindo", "Brogan Walker")
gamesSim(10000)
#iasmin decision
#winner correct, method wrong


# ### Jeremiah Wells vs Matthew Semelsberger
matchup_df = get_fighter_parameters("Jeremiah Wells", "Matthew Semelsberger")
gamesSim(10000)
#jeremiah decision

#------------------- >=15 swapped
#winner correct, method wrong

#--------------------------------- binomial
#winner wrong, 2nd option correct

#--------------------------------- binomial prop
#winner correct, method wrong

#----------------------------- binomial prop
#all wrong


# ### Rick Glenn vs Christos Giagos
matchup_df = get_fighter_parameters("Ricky Glenn", "Christos Giagos")
gamesSim(10000)
#christos ko

#---------------------------- >= 15 swapped
#winner correct, method wrong

#---------------------------- >= binomial
#winner correct, method wrong

#--------------------------------- binomial
#winner correct, method wrong

#----------------------------- binomial prop
#winner correct, method wrong


# ### Rani Yahya vs Montel Jackson
matchup_df = get_fighter_parameters("Rani Yahya", "Montel Jackson")
gamesSim(10000)
#montel ko 

#----------------------------->= 15 swapped 
#winner correct, method wrong

#----------------------------->= binomial 
#winner wrong, 2nd option correct

#--------------------------------- binomial prop
#winner correct, method wrong

#----------------------------- binomial prop
#winner wrong, 2nd option correct


# ### Karol Rosa vs Norma Dumont
matchup_df = get_fighter_parameters("Karol Rosa", "Norma Dumont")
gamesSim(10000)
#norma decision

#------------------------------->= 15 swapped
#winner wrong, 2nd option correct

#------------------------------- binomial
#winner wrong, 2nd option correct

#--------------------------------- binomial prop
#winner wrong, 2nd option correct

#----------------------------- binomial prop
#winner wrong, 2nd option correct


# ### Mohammed Usman vs Junior Tafa - N/A (Junior Tafa's first fight recorded on fightmetric)

# ### Francis Marshall vs William Gomis
matchup_df = get_fighter_parameters("Francis Marshall", "William Gomis")
gamesSim(10000)
#william decision
#perfect

#----------------- binomial
#all wrong

#--------------------------------- binomial prop
#perfect

#----------------------------- binomial prop
#all wrong


# ### Brady Hiestand vs Batgerel Danaa
matchup_df = get_fighter_parameters("Brady Hiestand", "Batgerel Danaa")
gamesSim(10000)
#brady ko

#------------------------------ >= 15 swapped
#winner correct, method wrong

#------------------------------ binomial
#winner correct, method wrong

#--------------------------------- binomial prop
#winner correct, method wrong

#----------------------------- binomial prop
#winner correct, method wrong


# Total Results for this event


#-------------------------------------- >= 15 swapped
#perfect: 2/9 
#2nd accepted: 4/9
#winner: 7/9 winner correct

#--------------------------------------- binomial 
#perfect: 1/9 
#2nd accepted: 5/9
#winner: 5/9

#-------------------------------------- binomial prop 
#perfect: 2/9 perfect 
#2nd accepted: 4/9 
#winner: 7/9 


# ## Predictions on UFC Fight Night: Song vs Simón

# ### Song Yadong vs Ricky Simón
matchup_df = get_fighter_parameters("Song Yadong", "Ricky Simon")
gamesSim(10000)
#song ko

#winner wrong, 2nd option correct 


# ### Caio Borralho vs Michal Oleksiejczuk
matchup_df = get_fighter_parameters("Caio Borralho", "Michal Oleksiejczuk")
gamesSim(10000)
#caio submission

#----------------------------- >= 15 swapped
#winner correct, method wrong


# ### Rodolfo Vieira vs Cody Brundage
matchup_df = get_fighter_parameters("Rodolfo Vieira", "Cody Brundage")
gamesSim(10000)
#rodolfo submission
#perfect


# ### Julian Erosa vs Fernando Padilla - N/A (Fernando Padilla's first fight recorded on fightmetric)

# ### Marcos Rogério de Lima vs Waldo Cortes-Acosta
matchup_df = get_fighter_parameters("Marcos Rogerio de Lima", "Cody Brundage")
gamesSim(10000)
#marcos decision

#---------------------------- >=15 swapped
#winner correct, 3rd option correct


# ### Josh Quinlan vs Trey Waters
matchup_df = get_fighter_parameters("Josh Quinlan", "Trey Waters")
gamesSim(10000)
#trey decision
#all wrong 


# ### Martin Buday vs Jake Collier
matchup_df = get_fighter_parameters("Martin Buday", "Jake Collier")
gamesSim(10000)
#martin decision
#perfect


# ### Cody Durden vs Charles Johnson
matchup_df = get_fighter_parameters("Cody Durden", "Charles Johnson")
gamesSim(10000)
#cody decision
#perfect

#Final results for this event

#3/7 perfect
#4/7 if 2nd highest outcome is accepted
#5/7 winner correct


# ### Stephanie Egger vs Irina Alekseeva - N/A (Irina Alekseeva first fight recorded on fightmetric)

# ### Journey Newson vs Marcus McGhee - N/A (Marcus McGhee first fight recorded on fightmetric)

# ### Hailey Cowan vs Jamey-Lyn Horth - N/A (Jamey-Lyn Horth first fight recorded on fightmetric)

# ## Predictions on UFC 288: Sterling vs. Cejudo

# ### Aljamain Sterling vs Henry Cejudo
matchup_df = get_fighter_parameters("Aljamain Sterling", "Henry Cejudo")
gamesSim(10000)
#aljamain decision

#------------------- >=15 swapped
#perfect

#------------------- >=20 swapped
#all wrong, 2nd option correct

#------------------- binomial
#perfect

#--------------------------------- binomial prop
#winner wrong, 2nd option correct


# ### Belal Muhammad vs Gilbert Burns
matchup_df = get_fighter_parameters("Belal Muhammad", "Gilbert Burns")
gamesSim(10000)
#belal decision

#------------------- >=15 swapped
#perfect

#------------------- >=20 swapped
#all wrong, 2nd option correct

#------------------- binomial
#perfect

#--------------------------------- binomial prop
#perfect


# ### Jessica Andrade vs Yan Xiaonan
matchup_df = get_fighter_parameters("Jessica Andrade", "Yan Xiaonan")
gamesSim(10000)
#yan ko

#---------------------------- >= 15 swapped
#all wrong, method wrong

#---------------------------- >= 20 swapped
#all wrong, method wrong

#------------------------ binomial
#winner correct, method wrong

#--------------------------------- binomial prop
#winner correct, method wrong


# ### Movsar Evloev vs Bryce Mitchell
matchup_df = get_fighter_parameters("Movsar Evloev", "Bryce Mitchell")
gamesSim(10000)
#movsar decision
#perfect


# ### Kron Gracie vs Charles Jourdain
matchup_df = get_fighter_parameters("Kron Gracie", "Charles Jourdain")
gamesSim(10000)
#charles decision

#------------------>= 15 swapped
#perfect

#------------------>= 20 swapped
#all wrong, 2nd option correct

#------------------ binomial
#all wrong

#--------------------------------- binomial prop
#perfect
#summing percentages: winner wrong


# ### Drew Dober vs Matt Frevola
matchup_df = get_fighter_parameters("Drew Dober", "Matt Frevola")
gamesSim(10000)
#matt ko

#------------------ >= 15 swapped
#all wrong, method wrong

#------------------ >= 20 swapped
#all wrong, method wrong 

#---------------------------- binomial
#all wrong

#--------------------------------- binomial prop
#winner wrong, method wrong


# ### Kennedy Nzechukwu vs Devin Clark
matchup_df = get_fighter_parameters("Kennedy Nzechukwu", "Devin Clark")
gamesSim(10000)
#kennedy submission

#------------------------------ >= 15 swapped
#winner wrong, 3rd option correct

#------------------------------ >= 20 swapped
#winner correct, method wrong

#------------------------------ binomial
#winner correct, method wrong

#--------------------------------- binomial prop
#winner correct, method wrong


# ### Khaos Williams vs Rolando Bedoya - N/A (Rolando Bedoya first fight recorded on fightmetric)

# ### Marina Rodriguez vs Virna Jandiroba
matchup_df = get_fighter_parameters("Marina Rodriguez", "Virna Jandiroba")
gamesSim(10000)
#virna decision

#--------------------- >= 15 swapped
#all wrong, method wrong

#--------------------- >= 20 swapped
#all wrong, method wrong

#--------------------- binomial 
#winner wrong, 2nd option correct

#--------------------------------- binomial prop
#all wrong


# ### Braxton Smith vs Parker Porter - N/A (Braxton Smith first fight recorded on fightmetric)

# ### Phil Hawes vs Ikram Aliskerov
matchup_df = get_fighter_parameters("Phil Hawes", "Ikram Aliskerov")
gamesSim(10000)
#ikram ko

#---------------------------- >= 15 swapped
#winner correct, 3rd option correct

#---------------------------- >= 20 swapped
#winner correct, 2nd option correct

#---------------------------- binomial
#winner wrong, 2nd option correct

#--------------------------------- binomial prop
#winner correct, method wrong


# ### Rafael Estevam vs Zhalgas Zhumagulov - Cancelled

# ### Joseph Holmes vs Claudio Ribeiro
matchup_df = get_fighter_parameters("Joseph Holmes", "Claudio Ribeiro")
gamesSim(10000)
#claudio ko

#--------------------- >=15 swapped
#perfect

#--------------------- >=20 swapped
#perfect

#--------------------- binomial
#all wrong

#--------------------------------- binomial prop
#winner wrong, 2nd option correct


# ### Daniel Santos vs Johnny Munoz - Cancelled

#Final results for this event

#------------------------------- binomial
#perfect: 3/10
#2nd accepted: 5/10
#winner: 5/10

#--------------------------------- binomial prop
#3/10 perfect
#5/10 if 2nd option is accepted
#5/10 winner correct (summing)

#try as n gets larger mean np is paired with sd sqrt(npq) and compare to current and to binomial

#https://courses.lumenlearning.com/introstats1/chapter/a-population-proportion/





