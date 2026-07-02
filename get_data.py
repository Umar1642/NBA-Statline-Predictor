from datetime import datetime
import pandas as pd
import os
import glob

os.makedirs("Data", exist_ok=True)

current_year = datetime.now().year

if datetime.now().month >= 6:
    last_completed_season = current_year
else:
    last_completed_season = current_year - 1

existing_files = glob.glob("Data/*.csv")
existing_seasons = {
    os.path.splitext(os.path.basename(file))[0]
    for file in existing_files
}

start_year = 2020
end_year = last_completed_season
years = range(start_year, end_year + 1)

for year in years:
    season_name = f"{year - 1}-{str(year)[-2:]} Stats"

    if season_name in existing_seasons:
        print(f"{season_name} Data already exists")
        continue
    
    print(f"Downloading {season_name}...")
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"

    df = pd.read_html(url)[0]
    df = df[df["Player"] != "Player"]

    df["Season"] = season_name
    df["SeasonYear"] = year

    df.to_csv(f"Data/{season_name}.csv", index=False)

    print(f"Saved Data/{season_name}.csv")
