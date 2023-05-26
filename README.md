# 10,000 Monte Carlo Simulations to predict UFC fights

I will be conducting 10,000 monte carlo simulations per matchup and will attempt to predict its outcome. This includes the specific winner of the fight and the method in which the win was carried out whether that be through decision, KO/TKO, or any form of submission. 

## Notable Results

NOTE: Some fighters in these three events did not have pre-existing match history on fightmetric.com and hence predictions could not be made for their matchups. As a result, out of the total 33 fights that happened across these three events, I could only make predictions for 26 of them. 

**Prediction Summary:** 
Perfect (We got the winner correct and the method in which the winner won): 9/26 ≈ 34.6%

2nd Highest outcome included (Perfect predictions + Next highest probability outcome was correct): 9/26 + 4/26 = 13/26 = 50%

Correct winner only (We got the winner correct although the method in which the winner won was incorrect): 18/26 ≈ 69.2%

## Code and Resources Used
Python Version: 3.8

Packages: numpy, scipy, pandas, matplotlib, seaborn, tabulate

For Web Framework Requirements: pip install -r requirements.txt

## Web Scraping

I used [Browserflow](https://browserflow.app/) to grab all the matches on fightmetric.com. These are all the matches that have happened in UFC history. 

The elements that I scraped include: 

* Fighter Name
* Weight (xyz lbs) 
* Outcome (Win/Lose)
* Full Matchup (Fighter vs His Opponent)
* Round that the match ended in
* Time that the match ended in

## Data Cleaning
After scraping the data, I needed to clean it up so that it would be useable for some EDA and simulation. The following changes were made:

* Deleted the first row of every fighter in the dataframe as it did not contain any match details. 
* Deleted all matches for upcoming UFC events. These matches were labelled as "NEXT". 
* In the matchup column, it contained both fighter names. when we only wanted to keep only opponents name in matchup column
* Removed outcomes we don't want in our dataset. This includes the likes of No Contest (NC) or Draws.
* Removed methods we don't want in our dataset. This includes the likes of Overturned, Could Not Continue (CNC) or Other.
* Cleaned the weight column by removing the "weight" and 'lbs' text from every entry while mapping numeric weights to a weight class label (lightweight, heavyweight, etc). 
* Narrowed down the remaining outcomes to three easily recognizable win methods: Decision, KO/TKO and Submission. 

## Exploratory Data Analysis (EDA)

I wanted to answer the following questions: 

**What is the most common method of winning?** 
<p align="center">
<img src="https://github.com/charlez1998/Monte-Carlo-Projects/assets/37009618/55e7bc2a-d8ec-47c6-8e64-08b25f731dc5" width=75% height=75%>
</p>

**For fighters that win through KO/Submission what round does it typically occur in?**

  <tr>
    <td align="left"><img src="https://github.com/charlez1998/Monte-Carlo-Projects/assets/37009618/e333f291-ef12-4df0-a474-2128d8ec1992" width=48% height=48%></td>
    <td align="right"><img src="https://github.com/charlez1998/Monte-Carlo-Projects/assets/37009618/e0a9fec6-df85-4f16-8445-890638481938" width=48% height=48%></td>
  </tr>
  
**Is there any discrepancy in the way fighters win across different weight classes?**
<p align="center">
<img src="https://github.com/charlez1998/Monte-Carlo-Projects/assets/37009618/d0996566-48a5-4942-b764-1d16a8eb9c26" width=75% height=75%>
</p>

Here is a tabulated version of the stacked bar plot above: 

<p></p>

<table align="center">
  <tr>
    <th></th>
    <th>Decision</th>
    <th>KO/TKO</th>
    <th>Submission</th>
  </tr>
  <tr>
    <td>Strawweight</td>
    <td>61.87%</td>
    <td>15.33%</td>
    <td>22.80%</td>
  </tr>
  <tr>
    <td>Flyweight</td>
    <td>55.87%</td>
    <td>21.13%</td>
    <td>22.99%</td>
  </tr>
  <tr>
    <td>Bantamweight</td>
    <td>53.20%</td>
    <td>27.17%</td>
    <td>19.62%</td>
  </tr>
  <tr>
    <td>Featherweight</td>
    <td>50.85%</td>
    <td>28.35%</td>
    <td>20.80%</td>
  </tr>
  <tr>
    <td>Lightweight</td>
    <td>46.17%</td>
    <td>29.13%</td>
    <td>24.70%</td>
  </tr>
  <tr>
    <td>Welterweight</td>
    <td>44.91%</td>
    <td>33.39%</td>
    <td>21.70%</td>
  </tr>
  <tr>
    <td>Middleweight</td>
    <td>36.60%</td>
    <td>39.79%</td>
    <td>23.60%</td>
  </tr>
  <tr>
    <td>Light Heavyweight</td>
    <td>33.15%</td>
    <td>46.24%</td>
    <td>20.61%</td>
  </tr>
  <tr>
    <td>Heavyweight</td>
    <td>25.21%</td>
    <td>52.24%</td>
    <td>22.55%</td>
  </tr>
</table>

<p></p>

Notice how the frequency of matches that end by a decision decreases while the frequency of matches that end with a knockout/tko increases as the weight class gets heavier.

# Simulation Results

Please refer to the notebook "UFC MC Sim Results" to view the specific simulation results across the three events: UFC Fight Night: Pavlovich vs Blaydes, UFC Fight Night: Song vs Simón and UFC 288: Sterling vs Cejudo
