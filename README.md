# 2025-26 NBA Statline Predictor

This project uses Gradient Boosting, specifically XGBRegressor from the XGBoost library, 
and predicts players statlines for the 2025-26 NBA season. This model uses historic data starting from the 2019-2020 NBA season all the way up to the 2024-2025 NBA Season. Specifically players points per game, rebounds per game, assists per game,
steals per game, and blocks per game. To use my program, run the streamlit front end, where you can search for a players name and find 
the players current seasons averages, their predicted seasons averages, a table with all their previous seasons averages, and a double line graph, that utilizes matplotlib, which shows their previous seasons points, predicted points, and their age.  

## Data 
The data is from:
- Basketball Reference


I automated the process of fetching all the seasons data starting from the 2019-20 NBA Season, all the way up to the current NBA Season. 

### Libraries needed
Here is a list of the libraries needed:
for the ML Model:
- xgboost (make sure to import XGBRegressor)
- joblib
- matplotlib.pyplot as plt
- glob 
- os
- sklearn.metrics (make sure to import mean_absolute_error)

for app.py:
- streamlit 
- pandas 
- joblib
- glob
- os
- matplotlib.pyplot 
- matplotlib.ticker 
- numpy 

for get_data.py:
- datetime
- pandas
- os
- glob


*** IMPORTANT ***
Make sure that you download the entire folder, since the CSV file is required for the prediction, the models are also included 
in the folder, and the streamlit front end is also included in the folder.
