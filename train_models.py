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

files = glob.glob("Data/*.csv")
df_list = []

# Goes through all the CSV files in the Data folder
for file in files:
    temp = pd.read_csv(file)
    season = os.path.basename(file).replace(".csv", "")
    temp["Season"] = season
    
    df_list.append(temp)
    

df = pd.concat(df_list, ignore_index=True)

df = df.copy()

tot_players = df[df["Team"] == "TOT"]["Player"].unique()

df = df[(df["Team"] == "TOT") | (~df["Player"].isin(tot_players))]

df = df[df["G"] >= 20]

df["SeasonYear"] = df["Season"].str[:4].astype(int)
df["AgeSquared"] = df["Age"] ** 2
df["Experience"] = (df.groupby("Player").cumcount())

df = df.sort_values(by = ["Player", "SeasonYear"]).reset_index(drop=True)

df["Target_PTS"] = (df.groupby("Player")["PTS"].shift(-1))
df["Target_TRB"] = (df.groupby("Player")["TRB"].shift(-1))
df["Target_AST"] = (df.groupby("Player")["AST"].shift(-1))
df["Target_STL"] = (df.groupby("Player")["STL"].shift(-1))
df["Target_BLK"] = (df.groupby("Player")["BLK"].shift(-1))

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

df["PTS_Rolling3"] = (df.groupby("Player")["PTS"].transform(lambda x: x.shift(1).rolling(3).mean()))
df["TRB_Rolling3"] = (df.groupby("Player")["TRB"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
df["AST_Rolling3"] = (df.groupby("Player")["AST"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
df["STL_Rolling3"] = (df.groupby("Player")["STL"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
df["BLK_Rolling3"] = (df.groupby("Player")["BLK"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
df["MP_Rolling3"] = (df.groupby("Player")["MP"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))


df = df.dropna(subset=["Target_PTS", "Target_TRB", "Target_AST", "Target_STL", "Target_BLK"])

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

# Train on seasons before 2023
train_df = df[df["SeasonYear"] < 2023]

# Test on 2023 and later
test_df = df[df["SeasonYear"] >= 2023]

#Training and testing all models
X_train_pts = train_df[points_features]
X_test_pts = test_df[points_features]

X_train_trb = train_df[rebound_features]
X_test_trb = test_df[rebound_features]

X_train_ast = train_df[assist_features]
X_test_ast = test_df[assist_features]

X_train_stl = train_df[steal_features]
X_test_stl = test_df[steal_features]

X_train_blk = train_df[block_features]
X_test_blk = test_df[block_features]

# Training targets
y_train_pts = train_df["Target_PTS"]
y_train_trb = train_df["Target_TRB"]
y_train_ast = train_df["Target_AST"]
y_train_stl = train_df["Target_STL"]
y_train_blk = train_df["Target_BLK"]

# Testing targets
y_test_pts = test_df["Target_PTS"]
y_test_trb = test_df["Target_TRB"]
y_test_ast = test_df["Target_AST"]
y_test_stl = test_df["Target_STL"]
y_test_blk = test_df["Target_BLK"]

# Create models
model_pts = XGBRegressor(**MODEL_PARAMS)

model_trb = XGBRegressor(**MODEL_PARAMS)

model_ast = XGBRegressor(**MODEL_PARAMS)

model_stl = XGBRegressor(**MODEL_PARAMS)

model_blk = XGBRegressor(**MODEL_PARAMS)

model_pts.fit(X_train_pts, y_train_pts)
model_trb.fit(X_train_trb, y_train_trb)
model_ast.fit(X_train_ast, y_train_ast)
model_stl.fit(X_train_stl, y_train_stl)
model_blk.fit(X_train_blk, y_train_blk)

# Saves models to disk
joblib.dump(model_pts, "Models/pts_model.pkl")
joblib.dump(model_trb, "Models/trb_model.pkl")
joblib.dump(model_ast, "Models/ast_model.pkl")
joblib.dump(model_stl, "Models/stl_model.pkl")
joblib.dump(model_blk, "Models/blk_model.pkl")


print(
    "PTS MAE:",
    mean_absolute_error(
        y_test_pts,
        model_pts.predict(X_test_pts)
    )
)

print(
    "TRB MAE:",
    mean_absolute_error(
        y_test_trb,
        model_trb.predict(X_test_trb)
    )
)

print(
    "AST MAE:",
    mean_absolute_error(
        y_test_ast,
        model_ast.predict(X_test_ast)
    )
)

print(
    "STL MAE:",
    mean_absolute_error(
        y_test_stl,
        model_stl.predict(X_test_stl)
    )
)

print(
    "BLK MAE:",
    mean_absolute_error(
        y_test_blk,
        model_blk.predict(X_test_blk)
    )
)
