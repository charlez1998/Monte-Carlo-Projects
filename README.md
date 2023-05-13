# Predictions for UFC Fight Night: Pavlovich vs Blaydes, UFC Fight Night: Song vs Simón and UFC 288: Sterling vs Cejudo

I will be conducting 10,000 monte carlo simulations per matchup and will attempt to predict its outcome. This includes the specific winner of the fight and the method in which the win was carried out whether that be through decision, KO/TKO, or any form of submission. 

## Notable Results

NOTE: Some fighters in these three events did not have pre-existing match history on fightmetric.com and hence predictions could not be made for their matchups. As a result, out of the total 33 fights that happened across these three events, I could only make predictions for 26 of them. 

(bold) Prediction Summary: 
Perfect (We got the winner correct and the method in which the winner won): 9/26 ≈ 34.6%
2nd Highest outcome included (Perfect predictions + Next highest probability outcome was correct): 9/26 + 4/26 = 13/26 = 50%
Correct winner only (We got the winner correct although the method in which the winner won was incorrect): 19/26 ≈ 73.1%

## Code and Resources Used
Python Version: 3.8

Packages: numpy, scipy, pandas, matplotlib, seaborn, tabulate

For Web Framework Requirements: pip install -r requirements.txt

## Web Scraping

I used [Browserflow](https://browserflow.app/) to grab all the matches on fightmetric.com. These are all the matches that have happened in UFC history. 

The elements that I scraped include: 

*Fighter Name
*Weight (xyz lbs) 
*Outcome (Win/Lose)
*Full Matchup (Fighter vs His Opponent)
*Round that the match ended in
*Time that the match ended in

## Data Cleaning
After scraping the data, I needed to clean it up so that it would be useable for some EDA and simulation. The following changes were made:

# delete first row of every fighter

#find matches that are upcoming and delete them

# delete "weight text" in weight column

# keep only opponents name in matchup column

#discovered there is a duplicate fighter name 


#types of outcomes we don't want in our dataset

#mapping weights to weight class label and assigning unknown to empty weight entries

#grouped specific outcomes to the three notable methods decision, ko/tko and submission

## Exploratory Data Analysis (EDA)
