import pandas as pd
from xgboost import XGBRegressor
import joblib
import glob
import os
from sklearn.metrics import mean_absolute_error


MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}

DATA_FOLDER = "Data"
MODEL_FOLDER = "Models"

os.makedirs(MODEL_FOLDER, exist_ok=True)


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



# Keeps the TOT row if a player was traded mid-season
tot_players = df[df["Team"] == "TOT"]["Player"].unique()

df = df[
    (df["Team"] == "TOT") |
    (~df["Player"].isin(tot_players))
]

# Only use players who played at least 20 games
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
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

df["TRB_Rolling3"] = (
    df.groupby("Player")["TRB"]
    .transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
)

df["AST_Rolling3"] = (
    df.groupby("Player")["AST"]
    .transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
)

df["STL_Rolling3"] = (
    df.groupby("Player")["STL"]
    .transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
)

df["BLK_Rolling3"] = (
    df.groupby("Player")["BLK"]
    .transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
)

df["MP_Rolling3"] = (
    df.groupby("Player")["MP"]
    .transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
)


df["Target_PTS"] = (
    df.groupby("Player")["PTS"].shift(-1)
)

df["Target_TRB"] = (
    df.groupby("Player")["TRB"].shift(-1)
)

df["Target_AST"] = (
    df.groupby("Player")["AST"].shift(-1)
)

df["Target_STL"] = (
    df.groupby("Player")["STL"].shift(-1)
)

df["Target_BLK"] = (
    df.groupby("Player")["BLK"].shift(-1)
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


training_df = df.dropna(
    subset=[
        "Target_PTS",
        "Target_TRB",
        "Target_AST",
        "Target_STL",
        "Target_BLK"
    ]
).copy()


train_df = training_df[
    training_df["SeasonYear"] < 2023
]

test_df = training_df[
    training_df["SeasonYear"] >= 2023
]



eval_pts_model = XGBRegressor(**MODEL_PARAMS)
eval_trb_model = XGBRegressor(**MODEL_PARAMS)
eval_ast_model = XGBRegressor(**MODEL_PARAMS)
eval_stl_model = XGBRegressor(**MODEL_PARAMS)
eval_blk_model = XGBRegressor(**MODEL_PARAMS)


eval_pts_model.fit(
    train_df[points_features],
    train_df["Target_PTS"]
)

eval_trb_model.fit(
    train_df[rebound_features],
    train_df["Target_TRB"]
)

eval_ast_model.fit(
    train_df[assist_features],
    train_df["Target_AST"]
)

eval_stl_model.fit(
    train_df[steal_features],
    train_df["Target_STL"]
)

eval_blk_model.fit(
    train_df[block_features],
    train_df["Target_BLK"]
)

print()
print("Historical model performance:")
print("--------------------------------")

print(
    "PTS MAE:",
    mean_absolute_error(
        test_df["Target_PTS"],
        eval_pts_model.predict(test_df[points_features])
    )
)

print(
    "TRB MAE:",
    mean_absolute_error(
        test_df["Target_TRB"],
        eval_trb_model.predict(test_df[rebound_features])
    )
)

print(
    "AST MAE:",
    mean_absolute_error(
        test_df["Target_AST"],
        eval_ast_model.predict(test_df[assist_features])
    )
)

print(
    "STL MAE:",
    mean_absolute_error(
        test_df["Target_STL"],
        eval_stl_model.predict(test_df[steal_features])
    )
)

print(
    "BLK MAE:",
    mean_absolute_error(
        test_df["Target_BLK"],
        eval_blk_model.predict(test_df[block_features])
    )
)

production_pts_model = XGBRegressor(**MODEL_PARAMS)
production_trb_model = XGBRegressor(**MODEL_PARAMS)
production_ast_model = XGBRegressor(**MODEL_PARAMS)
production_stl_model = XGBRegressor(**MODEL_PARAMS)
production_blk_model = XGBRegressor(**MODEL_PARAMS)


production_pts_model.fit(
    training_df[points_features],
    training_df["Target_PTS"]
)

production_trb_model.fit(
    training_df[rebound_features],
    training_df["Target_TRB"]
)

production_ast_model.fit(
    training_df[assist_features],
    training_df["Target_AST"]
)

production_stl_model.fit(
    training_df[steal_features],
    training_df["Target_STL"]
)

production_blk_model.fit(
    training_df[block_features],
    training_df["Target_BLK"]
)

joblib.dump(
    production_pts_model,
    os.path.join(MODEL_FOLDER, "pts_model.pkl")
)

joblib.dump(
    production_trb_model,
    os.path.join(MODEL_FOLDER, "trb_model.pkl")
)

joblib.dump(
    production_ast_model,
    os.path.join(MODEL_FOLDER, "ast_model.pkl")
)

joblib.dump(
    production_stl_model,
    os.path.join(MODEL_FOLDER, "stl_model.pkl")
)

joblib.dump(
    production_blk_model,
    os.path.join(MODEL_FOLDER, "blk_model.pkl")
)


latest_season = df["SeasonYear"].max()

print()
print("Production models trained successfully.")
print("----------------------------------------")
print(f"Latest season in data: {latest_season}")
print(f"Production training rows: {len(training_df)}")
print(f"Models saved to: {MODEL_FOLDER}")
