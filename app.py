import streamlit as st
import pandas as pd
import joblib
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Statline Predictor",
    page_icon="🏀",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0ede6;
}

.stApp {
    background-color: #0d0d0d;
}

/* ── Header ── */
.nba-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
}
.nba-header h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 0.12em;
    color: #f0ede6;
    margin: 0;
    line-height: 1;
}
.nba-header .accent {
    color: #f4572a;
}
.nba-header p {
    font-size: 0.95rem;
    color: #888;
    margin-top: 0.4rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* ── Search ── */
.stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 6px !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f4572a !important;
    box-shadow: 0 0 0 2px rgba(244,87,42,0.25) !important;
}
.stTextInput label {
    color: #888 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Stat Cards ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-card {
    background: #161616;
    border: 1px solid #252525;
    border-radius: 8px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: #f4572a;
}
.stat-card.predicted::before {
    background: #3a8cf5;
}
.stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.4rem;
}
.stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #f0ede6;
    line-height: 1;
}
.stat-delta {
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 0.35rem;
}
.delta-pos { color: #4caf82; }
.delta-neg { color: #e05252; }
.delta-neu { color: #888; }

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label span {
    display: inline-block;
    width: 28px;
    height: 2px;
    background: #f4572a;
}
.section-label.blue span { background: #3a8cf5; }

/* ── Player info bar ── */
.player-bar {
    display: flex;
    align-items: baseline;
    gap: 1.2rem;
    margin-bottom: 0.3rem;
}
.player-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: 0.06em;
    color: #f0ede6;
    line-height: 1;
}
.player-meta {
    font-size: 0.85rem;
    color: #666;
    font-weight: 300;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
}

/* ── Suggestion list ── */
.suggestion-item {
    padding: 0.5rem 0.8rem;
    cursor: pointer;
    border-radius: 4px;
    color: #bbb;
    font-size: 0.9rem;
}

/* ── No result ── */
.no-result {
    text-align: center;
    padding: 4rem 0;
    color: #444;
    font-size: 1rem;
    letter-spacing: 0.04em;
}

/* ── Matplotlib dark ── */
.element-container { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    files = glob.glob("Data/*.csv")
    if not files:
        return pd.DataFrame()
    df_list = []
    for file in files:
        temp = pd.read_csv(file)
        season = os.path.basename(file).replace(".csv", "")
        temp["Season"] = season
        df_list.append(temp)
    df = pd.concat(df_list, ignore_index=True)

    tot_players = df[df["Team"] == "TOT"]["Player"].unique()
    df = df[(df["Team"] == "TOT") | (~df["Player"].isin(tot_players))]
    df = df[df["G"] >= 20]

    df["SeasonYear"] = df["Season"].str[:4].astype(int)
    df["AgeSquared"] = df["Age"] ** 2
    df = df.sort_values(["Player", "SeasonYear"]).reset_index(drop=True)

    # Lag / trend features
    for stat in ["PTS", "TRB", "AST", "STL", "BLK"]:
        df[f"Prev_{stat}"] = df.groupby("Player")[stat].shift(1)
        df[f"{stat}_Change"] = df[stat] - df[f"Prev_{stat}"]

    df["PTS_Rolling3"] = df.groupby("Player")["PTS"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    trend_cols = [c for c in df.columns if c.startswith("Prev_") or c.endswith("_Change")]
    df[trend_cols] = df[trend_cols].fillna(0)

    return df


@st.cache_resource
def load_models():
    models = {}
    for stat in ["pts", "trb", "ast", "stl", "blk"]:
        path = f"Models/{stat}_model.pkl"
        if os.path.exists(path):
            models[stat] = joblib.load(path)
    return models


FEATURES = [
    "AgeSquared", "G", "GS", "MP",
    "FG", "FGA", "FG%", "3P", "3PA", "3P%",
    "FT", "FTA", "FT%",
    "TRB", "DRB", "ORB", "AST", "STL", "BLK", "TOV", "PTS",
    "Prev_PTS", "Prev_TRB", "Prev_AST", "Prev_STL", "Prev_BLK",
    "PTS_Change", "TRB_Change", "AST_Change", "STL_Change", "BLK_Change",
    "PTS_Rolling3",
]


def predict_row(models, row_df):
    results = {}
    for stat, model in models.items():
        results[stat] = round(float(model.predict(row_df[FEATURES])[0]), 1)
    return results


def delta_html(current, predicted):
    diff = predicted - current
    if diff > 0.05:
        return f'<div class="stat-delta delta-pos">▲ +{diff:.1f}</div>'
    elif diff < -0.05:
        return f'<div class="stat-delta delta-neg">▼ {diff:.1f}</div>'
    else:
        return f'<div class="stat-delta delta-neu">— {diff:+.1f}</div>'


def ppg_chart(history_df, predicted_pts, player_name):
    """Return a matplotlib figure: dual-axis PPG history + age."""
    seasons = list(history_df["SeasonYear"].astype(str))
    ppg = list(history_df["PTS"])
    ages = list(history_df["Age"])

    is_active = predicted_pts is not None

    # Append prediction point only for active players
    if is_active:
        next_year = str(history_df["SeasonYear"].max() + 1)
        seasons.append(next_year)
        ppg.append(predicted_pts)
        ages.append(ages[-1] + 1)

    x = np.arange(len(seasons))

    fig, ax1 = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d0d0d")
    ax1.set_facecolor("#0d0d0d")

    if is_active:
        # PPG line (actual portion)
        ax1.plot(x[:-1], ppg[:-1], color="#f4572a", linewidth=2.5,
                 marker="o", markersize=5, zorder=3, label="PPG (actual)")
        # Predicted extension
        ax1.plot([x[-2], x[-1]], [ppg[-2], ppg[-1]], color="#3a8cf5",
                 linewidth=2, linestyle="--", marker="o", markersize=7,
                 zorder=3, label="PPG (predicted)")
        ax1.fill_between(x[:-1], ppg[:-1], alpha=0.08, color="#f4572a")
    else:
        # Retired — just draw the full actual history
        ax1.plot(x, ppg, color="#f4572a", linewidth=2.5,
                 marker="o", markersize=5, zorder=3, label="PPG (actual)")
        ax1.fill_between(x, ppg, alpha=0.08, color="#f4572a")

    ax1.set_ylabel("Points Per Game", color="#888", fontsize=9, labelpad=10)
    ax1.tick_params(axis="y", colors="#555", labelsize=8)
    ax1.tick_params(axis="x", colors="#555", labelsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seasons, rotation=30, ha="right")
    ax1.spines[["top", "right", "left", "bottom"]].set_color("#222")
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Age line on secondary axis
    ax2 = ax1.twinx()
    ax2.set_facecolor("#0d0d0d")
    ax2.plot(x, ages, color="#a07cf5", linewidth=1.5, linestyle=":",
             marker="s", markersize=4, alpha=0.7, label="Age")
    ax2.set_ylabel("Age", color="#a07cf5", fontsize=9, labelpad=10)
    ax2.tick_params(axis="y", colors="#a07cf5", labelsize=8)
    ax2.spines[["top", "right", "left", "bottom"]].set_color("#222")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", framealpha=0, fontsize=8,
               labelcolor="#999")

    fig.tight_layout(pad=1.5)
    return fig


# ── App layout ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nba-header">
  <h1>NBA <span class="accent">STATLINE</span> PREDICTOR</h1>
  <p>XGBoost · 2019–20 → 2024–25 seasons · predicts next-season per-game averages</p>
</div>
""", unsafe_allow_html=True)

df = load_data()
models = load_models()

if df.empty:
    st.error("⚠️ No CSV files found in the `Data/` folder. Make sure your season CSVs are there.")
    st.stop()

if not models:
    st.error("⚠️ No trained model files found in `Models/`. Run `train.py` first.")
    st.stop()

# Search box
search = st.text_input("PLAYER NAME", placeholder="e.g. LeBron James, Nikola Jokić…")

if not search.strip():
    st.markdown('<div class="no-result">Search for a player above to see their stats and prediction.</div>',
                unsafe_allow_html=True)
    st.stop()

# ── Match players ──────────────────────────────────────────────────────────────
all_players = sorted(df["Player"].unique())
query = search.strip().lower()
matches = [p for p in all_players if query in p.lower()]

if not matches:
    st.markdown(f'<div class="no-result">No player found matching <b>"{search}"</b>.</div>',
                unsafe_allow_html=True)
    st.stop()

# If multiple matches, let user pick
if len(matches) > 1:
    player_name = st.selectbox("Multiple players found — select one:", matches)
else:
    player_name = matches[0]

# ── Get player data ────────────────────────────────────────────────────────────
player_df = df[df["Player"] == player_name].sort_values("SeasonYear")
latest = player_df.iloc[-1]

# ── Retired check ──────────────────────────────────────────────────────────────
latest_season_in_dataset = df["SeasonYear"].max()
is_retired = int(latest["SeasonYear"]) < latest_season_in_dataset - 1

# ── Predict (only for active players) ─────────────────────────────────────────
preds = None if is_retired else predict_row(models, pd.DataFrame([latest]))

# ── Player name + meta ─────────────────────────────────────────────────────────
retired_badge = ' · <span style="color:#f4572a;font-size:0.75rem;letter-spacing:0.1em;">RETIRED</span>' if is_retired else ''
st.markdown(f"""
<div class="player-bar">
  <div class="player-name">{player_name}</div>
  <div class="player-meta">Age {int(latest['Age'])} · Last season: {latest['Season']} · {int(latest['G'])} G{retired_badge}</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Current season stats ───────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span></span>Current Season Averages</div>',
            unsafe_allow_html=True)

stats_map = {
    "PTS": ("Points", latest["PTS"]),
    "TRB": ("Rebounds", latest["TRB"]),
    "AST": ("Assists", latest["AST"]),
    "STL": ("Steals", latest["STL"]),
    "BLK": ("Blocks", latest["BLK"]),
}

cols = st.columns(5)
for col, (key, (label, val)) in zip(cols, stats_map.items()):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{val:.1f}</div>
        </div>""", unsafe_allow_html=True)

# ── Predicted stats ────────────────────────────────────────────────────────────
st.markdown('<br><div class="section-label blue"><span></span>Predicted Next Season</div>',
            unsafe_allow_html=True)

pred_map = {
    "pts": ("Points", latest["PTS"]),
    "trb": ("Rebounds", latest["TRB"]),
    "ast": ("Assists", latest["AST"]),
    "stl": ("Steals", latest["STL"]),
    "blk": ("Blocks", latest["BLK"]),
}

cols2 = st.columns(5)
if is_retired:
    for col, (key, (label, _)) in zip(cols2, pred_map.items()):
        with col:
            st.markdown(f"""
            <div class="stat-card predicted">
                <div class="stat-label">{label}</div>
                <div class="stat-value" style="font-size:1.6rem;color:#444;">N/A</div>
                <div class="stat-delta delta-neu">Retired</div>
            </div>""", unsafe_allow_html=True)
else:
    for col, (key, (label, current)) in zip(cols2, pred_map.items()):
        predicted = preds[key]
        with col:
            st.markdown(f"""
            <div class="stat-card predicted">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{predicted:.1f}</div>
                {delta_html(current, predicted)}
            </div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── PPG + Age chart ────────────────────────────────────────────────────────────
st.markdown('<div class="section-label"><span></span>Scoring History & Age Curve</div>',
            unsafe_allow_html=True)

fig = ppg_chart(player_df, preds["pts"] if not is_retired else None, player_name)
st.pyplot(fig, use_container_width=True)

# ── Career table (expandable) ──────────────────────────────────────────────────
with st.expander("📋 Full career stats table"):
    display_cols = ["Season", "Age", "Team", "G", "MP", "PTS", "TRB", "AST", "STL", "BLK",
                    "FG%", "3P%", "FT%"]
    available = [c for c in display_cols if c in player_df.columns]
    st.dataframe(
        player_df[available].set_index("Season").style
        .format(precision=1)
        .set_properties(**{"background-color": "#161616", "color": "#f0ede6"}),
        use_container_width=True,
    )
