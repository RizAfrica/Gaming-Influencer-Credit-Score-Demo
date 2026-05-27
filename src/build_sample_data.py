"""Build artifacts/sample_games.parquet from the raw Lichess PGN.

Run once from the repo root:
    python src/build_sample_data.py

The output file is used by the Streamlit app's "sample player" tab.
"""
import os
import sys
import pandas as pd
import chess.pgn
from datetime import datetime
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PGN_PATH = "/home/selam/Downloads/lichess_data/extracted/lichess_elite_2024-04.pgn"
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
MIN_GAMES = 20

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
out_path = os.path.join(ARTIFACTS_DIR, "sample_games.parquet")

if os.path.exists(out_path):
    print(f"Already exists: {out_path}")
    print("Delete it manually if you want to rebuild.")
    sys.exit(0)

if not os.path.exists(PGN_PATH):
    print(f"PGN file not found at: {PGN_PATH}")
    print("Update PGN_PATH at the top of this script to point to your file.")
    sys.exit(1)

print(f"Parsing {PGN_PATH}...")
records = []
with open(PGN_PATH) as f:
    pbar = tqdm(desc="Parsing games")
    while True:
        try:
            game = chess.pgn.read_game(f)
        except Exception:
            continue
        if game is None:
            break
        h = game.headers
        white = h.get("White")
        black = h.get("Black")
        if not white or not black:
            pbar.update(1)
            continue

        try:
            we = int(h.get("WhiteElo", 0)) or None
            be = int(h.get("BlackElo", 0)) or None
        except ValueError:
            we = be = None
        try:
            wd = int(h.get("WhiteRatingDiff", 0))
            bd = int(h.get("BlackRatingDiff", 0))
        except ValueError:
            wd = bd = 0

        result = h.get("Result", "*")
        if result == "1-0":
            wo, bo = "win", "loss"
        elif result == "0-1":
            wo, bo = "loss", "win"
        elif result == "1/2-1/2":
            wo, bo = "draw", "draw"
        else:
            wo, bo = "unknown", "unknown"

        try:
            ts = datetime.strptime(
                h.get("UTCDate", "") + " " + h.get("UTCTime", ""),
                "%Y.%m.%d %H:%M:%S"
            )
        except ValueError:
            ts = None

        common = {
            "termination": h.get("Termination", "Unknown"),
            "time_control": h.get("TimeControl", ""),
            "opening": h.get("Opening", ""),
            "eco": h.get("ECO", ""),
            "timestamp": ts,
        }
        records.append({"player": white, "elo": we, "rating_diff": wd,
                        "color": "white", "outcome": wo, **common})
        records.append({"player": black, "elo": be, "rating_diff": bd,
                        "color": "black", "outcome": bo, **common})
        pbar.update(1)
    pbar.close()

games_df = pd.DataFrame(records)
print(f"\nParsed {len(games_df):,} game records")

games_per_player = games_df.groupby("player").size()
active = games_per_player[games_per_player >= MIN_GAMES].index
df_active = games_df[games_df["player"].isin(active)].copy()
print(f"Active players (>= {MIN_GAMES} games): {len(active):,}")

players_summary = df_active.groupby("player")["rating_diff"].sum().reset_index()
players_summary.columns = ["player", "rating_diff_total"]

high_risk = players_summary.nsmallest(20, "rating_diff_total")["player"].tolist()
low_risk = players_summary.nlargest(20, "rating_diff_total")["player"].tolist()
random_pick = players_summary.sample(20, random_state=42)["player"].tolist()
sample_names = list(set(high_risk + low_risk + random_pick))

sample_games = df_active[df_active["player"].isin(sample_names)].copy()
sample_games["timestamp"] = pd.to_datetime(sample_games["timestamp"])
sample_games.to_parquet(out_path)

print(f"\nSaved {len(sample_games):,} records "
      f"for {sample_games['player'].nunique()} players → {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1e6:.1f} MB")