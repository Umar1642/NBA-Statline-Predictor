import pandas as pd
import joblib
import glob
import os

DATA_FOLDER = "Data"
MODEL_FOLDER = "Models"
PREDICTION_FOLDER = "Predictions"

os.makedirs(PREDICTION_FOLDER, exist_ok=True)

files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

if not files:
    raise FileNotFoundError("No CSV files were found in the Data folder")

df_list = []

for file in files:
    temp = pd.read_csv(file)
    season = os.path.basename(file).replace(".csv", "")
    temp["Season"] = season

    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

df = df.copy()

#keeps the players TOT row, if they were traded mid-season
tot_players = df[df["Team"] == "TOT"]["Player"].unique()

df = df[
    (df["Team"] == "TOT") |
    (~df["Player"].isin(tot_players))
]

df = df[df["G"] >= 20]

df["SeasonYear"] = df["Season"].str[:4].astype(int)

df = df.sort_values(
    by=["Player", "SeasonYear"]
).reset_index(drop=True)

df["AgeSquared"] = df["Age"] ** 2

df["Experience"] = df.groupby("Player").cumcount()


df["PTS_Lag1"] = df.groupby("Player")["PTS"].shift(1)
df["TRB_Lag1"] = df.groupby("Player")["TRB"].shift(1)
df["AST_Lag1"] = df.groupby("Player")["AST"].shift(1)
df["STL_Lag1"] = df.groupby("Player")["STL"].shift(1)
df["BLK_Lag1"] = df.groupby("Player")["BLK"].shift(1)

df["MP_Lag1"] = df.groupby("Player")["MP"].shift(1)
df["G_Lag1"] = df.groupby("Player")["G"].shift(1)

df["PTS_Lag2"] = df.groupby("Player")["PTS"].shift(2)
df["TRB_Lag2"] = df.groupby("Player")["TRB"].shift(2)
df["AST_Lag2"] = df.groupby("Player")["AST"].shift(2)
df["STL_Lag2"] = df.groupby("Player")["STL"].shift(2)
df["BLK_Lag2"] = df.groupby("Player")["BLK"].shift(2)

df["MP_Lag2"] = df.groupby("Player")["MP"].shift(2)


df["PTS_Change"] = df["PTS"] - df["PTS_Lag1"]
df["TRB_Change"] = df["TRB"] - df["TRB_Lag1"]
df["AST_Change"] = df["AST"] - df["AST_Lag1"]
df["STL_Change"] = df["STL"] - df["STL_Lag1"]
df["BLK_Change"] = df["BLK"] - df["BLK_Lag1"]

df["MP_Change"] = df["MP"] - df["MP_Lag1"]

df["PTS_Rolling3"] = (
    df.groupby("Player")["PTS"]
    .transform(lambda x: x.shift(1).rolling(3).mean())
)

df["TRB_Rolling3"] = (
    df.groupby("Player")["TRB"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

df["AST_Rolling3"] = (
    df.groupby("Player")["AST"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

df["STL_Rolling3"] = (
    df.groupby("Player")["STL"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

df["BLK_Rolling3"] = (
    df.groupby("Player")["BLK"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

df["MP_Rolling3"] = (
    df.groupby("Player")["MP"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)


trend_cols = [
    "PTS_Lag1",
    "PTS_Lag2",
    "TRB_Lag1",
    "TRB_Lag2",
    "AST_Lag1",
    "AST_Lag2",
    "STL_Lag1",
    "STL_Lag2",
    "BLK_Lag1",
    "BLK_Lag2",
    "MP_Lag1",
    "MP_Lag2",
    "PTS_Change",
    "TRB_Change",
    "AST_Change",
    "STL_Change",
    "BLK_Change",
    "MP_Change",
    "PTS_Rolling3",
    "TRB_Rolling3",
    "AST_Rolling3",
    "STL_Rolling3",
    "BLK_Rolling3",
    "MP_Rolling3"
]

df[trend_cols] = df[trend_cols].fillna(0)

latest_season_year = df["SeasonYear"].max()

prediction_df = df[
    df["SeasonYear"] == latest_season_year
].copy()

if prediction_df.empty:
    raise ValueError("No players found for the latest season")


points_features = [
    "Age",
    "AgeSquared",
    "Experience",
    "MP",
    "PTS",
    "FG",
    "FGA",
    "FG%",
    "3P",
    "3PA",
    "3P%",
    "FT",
    "FTA",
    "FT%",
    "PTS_Lag1",
    "PTS_Lag2",
    "PTS_Rolling3",
    "PTS_Change",
    "MP_Lag1",
    "MP_Lag2",
    "MP_Rolling3",
    "MP_Change"
]


rebound_features = [
    "Age",
    "AgeSquared",
    "Experience",
    "MP",
    "TRB",
    "ORB",
    "DRB",
    "TRB_Lag1",
    "TRB_Lag2",
    "TRB_Rolling3",
    "TRB_Change",
    "MP_Lag1",
    "MP_Lag2",
    "MP_Rolling3",
    "MP_Change"
]

assist_features = [
    "Age",
    "AgeSquared",
    "Experience",
    "MP",
    "AST",
    "TOV",
    "AST_Lag1",
    "AST_Lag2",
    "AST_Rolling3",
    "AST_Change",
    "MP_Lag1",
    "MP_Lag2",
    "MP_Rolling3",
    "MP_Change"
]

steal_features = [
    "Age",
    "AgeSquared",
    "Experience",
    "MP",
    "STL",
    "STL_Lag1",
    "STL_Lag2",
    "STL_Rolling3",
    "STL_Change",
    "MP_Lag1",
    "MP_Lag2",
    "MP_Rolling3",
    "MP_Change"
]

block_features = [
    "Age",
    "AgeSquared",
    "Experience",
    "MP",
    "BLK",
    "BLK_Lag1",
    "BLK_Lag2",
    "BLK_Rolling3",
    "BLK_Change",
    "MP_Lag1",
    "MP_Lag2",
    "MP_Rolling3",
    "MP_Change"
]


pts_model = joblib.load(
    os.path.join(MODEL_FOLDER, "pts_model.pkl")
)

trb_model = joblib.load(
    os.path.join(MODEL_FOLDER, "trb_model.pkl")
)

ast_model = joblib.load(
    os.path.join(MODEL_FOLDER, "ast_model.pkl")
)

stl_model = joblib.load(
    os.path.join(MODEL_FOLDER, "stl_model.pkl")
)

blk_model = joblib.load(
    os.path.join(MODEL_FOLDER, "blk_model.pkl")
)


prediction_df["Predicted_PTS"] = pts_model.predict(
    prediction_df[points_features]
)

prediction_df["Predicted_TRB"] = trb_model.predict(
    prediction_df[rebound_features]
)

prediction_df["Predicted_AST"] = ast_model.predict(
    prediction_df[assist_features]
)

prediction_df["Predicted_STL"] = stl_model.predict(
    prediction_df[steal_features]
)

prediction_df["Predicted_BLK"] = blk_model.predict(
    prediction_df[block_features]
)


prediction_df["Predicted_PTS"] = prediction_df["Predicted_PTS"].round(1)
prediction_df["Predicted_TRB"] = prediction_df["Predicted_TRB"].round(1)
prediction_df["Predicted_AST"] = prediction_df["Predicted_AST"].round(1)
prediction_df["Predicted_STL"] = prediction_df["Predicted_STL"].round(1)
prediction_df["Predicted_BLK"] = prediction_df["Predicted_BLK"].round(1)


next_season_year = latest_season_year + 1

#gives 2019-20
next_season = (
    f"{next_season_year - 1}-"
    f"{str(next_season_year)[-2:]}"
)


output_cols = [
    "Player",
    "Team",
    "Season",
    "PTS",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "Predicted_PTS",
    "Predicted_TRB",
    "Predicted_AST",
    "Predicted_STL",
    "Predicted_BLK"
]

output_df = prediction_df[output_cols].copy()

output_file = os.path.join(PREDICTION_FOLDER, f"{next_season}_predictions.csv")

output_df.to_csv(
    output_file,
    index=False
)

print()
print("Prediction completed successfully")
print(f"Latest season: {prediction_df['Season'].iloc[0]}")
print(f"Prediction season: {next_season}")
print(f"Players predicted: {len(output_df)}")
print(f"Output file: {output_file}")