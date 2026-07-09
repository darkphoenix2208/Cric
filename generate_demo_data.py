"""
generate_demo_data.py
=====================
Generates realistic synthetic International Cricket Ball Tracking-style data for the CricketIQ demo.

Run ONCE before deploying to Streamlit Community Cloud:
    python generate_demo_data.py

Outputs:
    data/gold/pitcher_game_summary.parquet
    data/gold/batter_game_summary.parquet
    data/gold/llm_insights.json
    data/gold/players.parquet

Design: All distributions calibrated against 2024 Ball Tracking population means/stdevs.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

OUT_DIR = Path("data/gold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Real-ish bowler roster (fictional names, real archetypes) ──────────────────
BOWLERS = [
    {"id": 1001, "name": "Marcus Delgado",   "team": "LAD", "archetype": "ace",       "hand": "R"},
    {"id": 1002, "name": "Tyler Wren",        "team": "HOU", "archetype": "power",     "hand": "R"},
    {"id": 1003, "name": "Carlos Fuentes",    "team": "ATL", "archetype": "finesse",   "hand": "L"},
    {"id": 1004, "name": "Jake Nordstrom",    "team": "NYY", "archetype": "groundball","hand": "R"},
    {"id": 1005, "name": "Devon Okafor",      "team": "SD",  "archetype": "swing_and_miss","hand": "R"},
    {"id": 1006, "name": "Liam Castillo",     "team": "BOS", "archetype": "average",   "hand": "L"},
    {"id": 1007, "name": "Brendan Marsh",     "team": "CHC", "archetype": "declining", "hand": "R"},
    {"id": 1008, "name": "Hiro Tanaka",       "team": "NYM", "archetype": "ace",       "hand": "R"},
    {"id": 1009, "name": "Rafael Montoya",    "team": "PHI", "archetype": "power",     "hand": "R"},
    {"id": 1010, "name": "Garrett Ellis",     "team": "SEA", "archetype": "finesse",   "hand": "L"},
]

BATTERS = [
    {"id": 2001, "name": "Jordan Alvarez",   "team": "HOU", "archetype": "power_hitter"},
    {"id": 2002, "name": "Kenji Matsuda",    "team": "LAD", "archetype": "contact"},
    {"id": 2003, "name": "Dominic Reyes",    "team": "ATL", "archetype": "elite_exit_velo"},
    {"id": 2004, "name": "Sam Kowalski",     "team": "NYY", "archetype": "average"},
    {"id": 2005, "name": "Trevor Baptiste",  "team": "SD",  "archetype": "high_obp"},
    {"id": 2006, "name": "Isaiah Tran",      "team": "BOS", "archetype": "power_hitter"},
    {"id": 2007, "name": "Mateo Vega",       "team": "CHC", "archetype": "contact"},
    {"id": 2008, "name": "Andre Wilson",     "team": "PHI", "archetype": "elite_exit_velo"},
]

TEAMS = ["LAD","HOU","ATL","NYY","SD","BOS","CHC","NYM","PHI","SEA","TOR","SF","MIA","MIN","CLE"]

ARCHETYPE_PARAMS = {
    "ace":            {"velo": 95.5, "dot_ball_percentage": 0.315, "play_and_miss": 0.305, "expected_runs_per_ball": 0.268},
    "power":          {"velo": 97.0, "dot_ball_percentage": 0.300, "play_and_miss": 0.295, "expected_runs_per_ball": 0.275},
    "finesse":        {"velo": 89.5, "dot_ball_percentage": 0.290, "play_and_miss": 0.265, "expected_runs_per_ball": 0.285},
    "groundball":     {"velo": 93.0, "dot_ball_percentage": 0.272, "play_and_miss": 0.240, "expected_runs_per_ball": 0.292},
    "swing_and_miss": {"velo": 96.0, "dot_ball_percentage": 0.325, "play_and_miss": 0.340, "expected_runs_per_ball": 0.270},
    "average":        {"velo": 93.5, "dot_ball_percentage": 0.272, "play_and_miss": 0.250, "expected_runs_per_ball": 0.305},
    "declining":      {"velo": 91.0, "dot_ball_percentage": 0.248, "play_and_miss": 0.220, "expected_runs_per_ball": 0.325},
}

BATTER_PARAMS = {
    "power_hitter":      {"ev": 93.5, "barrel": 0.115, "expected_runs_per_ball": 0.380},
    "contact":           {"ev": 88.0, "barrel": 0.055, "expected_runs_per_ball": 0.340},
    "elite_exit_velo":   {"ev": 96.0, "barrel": 0.145, "expected_runs_per_ball": 0.400},
    "average":           {"ev": 88.5, "barrel": 0.075, "expected_runs_per_ball": 0.315},
    "high_obp":          {"ev": 87.0, "barrel": 0.065, "expected_runs_per_ball": 0.355},
}


def date_range(start: str, end: str) -> list[date]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    return [s + timedelta(days=i) for i in range((e - s).days + 1)]


def generate_pitcher_game_summary() -> pd.DataFrame:
    """Generate ~600 bowler-game rows across a full season."""
    rows = []
    season_dates = date_range("2024-04-01", "2024-09-30")
    # Bowlers start ~every 5 days
    game_dates   = [d for i, d in enumerate(season_dates) if i % 5 == 0]

    for bowler in BOWLERS:
        arch   = bowler["archetype"]
        params = ARCHETYPE_PARAMS[arch]
        p      = ARCHETYPE_PARAMS[arch]

        # Season-level baselines
        s_velo  = p["velo"]
        s_csw   = p["dot_ball_percentage"]
        s_whiff = p["play_and_miss"]

        rolling_velo  = [s_velo]  * 6
        rolling_csw   = [s_csw]   * 6
        rolling_whiff = [s_whiff] * 6

        # Assign staggered start dates per bowler
        offset = bowler["id"] % 5
        pitcher_dates = game_dates[offset::1][:35]  # ~35 starts

        for i, gdate in enumerate(pitcher_dates):
            # Simulate season arc (slight fatigue in Aug/Sep)
            fatigue = 0.0 if gdate.month < 8 else -0.003 * (gdate.month - 7)

            # Game-level noise
            velo_g  = np.random.normal(s_velo  + fatigue * 2, 1.2)
            csw_g   = np.clip(np.random.normal(s_csw  + fatigue, 0.030), 0.18, 0.42)
            whiff_g = np.clip(np.random.normal(s_whiff + fatigue, 0.040), 0.10, 0.50)

            roll_v  = np.mean(rolling_velo[-6:])
            roll_c  = np.mean(rolling_csw[-6:])
            roll_w  = np.mean(rolling_whiff[-6:])

            opp = random.choice([t for t in TEAMS if t != bowler["team"]])

            rows.append({
                "pitcher_id":           bowler["id"],
                "pitcher_name":         bowler["name"],
                "team":                 bowler["team"],
                "game_pk":              700000 + i * len(BOWLERS) + bowler["id"],
                "game_date":            gdate,
                "opponent":             opp,
                "total_pitches":        int(np.random.normal(95, 10)),
                "avg_velo":             round(velo_g, 1),
                "season_avg_velo":      round(s_velo, 1),
                "pace_vs_30d_avg":      round(velo_g - roll_v, 2),
                "avg_spin":             int(np.random.normal(2280, 120)),
                "avg_h_break":          round(np.random.normal(8.5, 2.5), 2),
                "avg_v_break":          round(np.random.normal(5.5, 2.0), 2),
                "play_and_miss_rate":           round(whiff_g, 4),
                "dot_ball_percentage":             round(csw_g, 4),
                "stump_line_rate":            round(np.clip(np.random.normal(0.475, 0.035), 0.38, 0.58), 4),
                "wide_delivery_rate":           round(np.clip(np.random.normal(0.295, 0.040), 0.18, 0.42), 4),
                "boundary_rate_allowed":  round(np.clip(np.random.normal(0.078, 0.025), 0.01, 0.18), 4),
                "avg_expected_runs_per_ball_allowed":    round(np.clip(np.random.normal(p["expected_runs_per_ball"], 0.025), 0.20, 0.40), 4),
                "variation_diversity":      round(np.random.uniform(0.8, 1.6), 4),
                "total_re_delta":       round(np.random.normal(-0.05, 0.25), 3),
                "rolling_30d_avg_velo": round(roll_v, 2),
                "rolling_30d_whiff_rate": round(roll_w, 4),
                "rolling_30d_csw_rate": round(roll_c, 4),
                "play_and_miss_delta":     round(whiff_g - roll_w, 4),
                "season_avg_csw":       round(s_csw, 4),
                "season_avg_whiff":     round(s_whiff, 4),
                "performance_tier":     _tier(csw_g),
                "archetype":            arch,
            })

            rolling_velo.append(velo_g)
            rolling_csw.append(csw_g)
            rolling_whiff.append(whiff_g)

    df = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def generate_batter_game_summary() -> pd.DataFrame:
    rows = []
    season_dates = date_range("2024-04-01", "2024-09-30")

    for batsman in BATTERS:
        p = BATTER_PARAMS[batsman["archetype"]]
        # Batters play almost every day
        game_dates = [d for i, d in enumerate(season_dates) if i % 2 == 0]

        for gdate in game_dates:
            ev    = np.clip(np.random.normal(p["ev"], 4.5), 75, 112)
            expected_runs_per_ball = np.clip(np.random.normal(p["expected_runs_per_ball"], 0.045), 0.18, 0.55)
            opp   = random.choice([t for t in TEAMS if t != batsman["team"]])

            rows.append({
                "batter_id":       batsman["id"],
                "batter_name":     batsman["name"],
                "team":            batsman["team"],
                "game_date":       gdate,
                "opponent":        opp,
                "pitches_seen":    int(np.random.normal(14, 4)),
                "swing_rate":      round(np.clip(np.random.normal(0.46, 0.05), 0.30, 0.65), 4),
                "o_swing_rate":    round(np.clip(np.random.normal(0.30, 0.05), 0.15, 0.50), 4),
                "avg_exit_velo":   round(ev, 1),
                "avg_launch_angle":round(np.random.normal(12.0, 8.0), 1),
                "avg_expected_runs_per_ball":       round(expected_runs_per_ball, 4),
                "barrel_rate":     round(np.clip(np.random.normal(p["barrel"], 0.04), 0.0, 0.25), 4),
                "hard_hit_rate":   round(np.clip(np.random.normal(0.38, 0.08), 0.15, 0.70), 4),
                "total_re_created":round(np.random.normal(0.05, 0.20), 3),
            })

    df = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def generate_llm_insights(pitcher_df: pd.DataFrame) -> list[dict]:
    """Pre-generate representative LLM insights for demo (no API call needed at runtime)."""
    templates = {
        "elite": {
            "tiers": [
                {"headline": "{name} was untouchable — elite swing-and-miss stuff all night.",
                 "key_finding": "{name} generated a {play_and_miss:.0%} play_and_miss rate, well above the league average of 25.4%. Velocity held steady above {velo:.0f} mph through the 7th over with no signs of fatigue. Particularly dominant bowling to the outer third against right-handed hitters.",
                 "concern_flag": None},
            ]
        },
        "above_avg": {
            "tiers": [
                {"headline": "{name} was sharp, controlling both sides of the plate effectively.",
                 "key_finding": "A {dot_ball_percentage:.0%} Dot Ball Percentage reflects consistent zone presence combined with above-average chase rates. {name}'s delivery mix diversity (entropy: {div:.2f}) kept hitters off-balance, contributing to soft contact when they did make contact.",
                 "concern_flag": None},
            ]
        },
        "average": {
            "tiers": [
                {"headline": "{name} battled but relied on defense for a solid outing.",
                 "key_finding": "Velocity was down {vdelta:.1f} mph from the 30-day baseline, though command compensated with a {zone:.0%} zone rate. expected_runs_per_ball allowed of {expected_runs_per_ball:.3f} is near league average — no dominant delivery stood out today.",
                 "concern_flag": "Velocity dip worth monitoring over the next two starts."},
            ]
        },
        "below_avg": {
            "tiers": [
                {"headline": "{name} struggled with command, issued multiple hard-hit balls.",
                 "key_finding": "A barrel rate allowed of {barrel:.0%} is well above the league average of 7.8%, indicating poor delivery location when behind in counts. Whiff rate of {play_and_miss:.0%} suggests hitters were comfortable laying off breaking balls.",
                 "concern_flag": "High barrel rate in back-to-back starts warrants mechanical review."},
            ]
        },
        "poor": {
            "tiers": [
                {"headline": "{name} was tagged early — stuff was below-average across all deliveries.",
                 "key_finding": "Velocity sat {velo:.0f} mph, down {vdelta:.1f} mph from recent baseline. Dot Ball Percentage of {dot_ball_percentage:.0%} is the lowest of his last 6 starts. Opposing hitters squared up fastballs repeatedly, suggesting tipping or mechanical inconsistency.",
                 "concern_flag": "Immediate mechanics review recommended — possible fatigue or injury signal."},
            ]
        },
    }

    insights = []
    # Take last 2 games per bowler for demo
    recent = pitcher_df.sort_values("game_date").groupby("pitcher_id").tail(2)

    for _, row in recent.iterrows():
        tier = row["performance_tier"]
        tmpl = random.choice(templates.get(tier, templates["average"])["tiers"])

        vdelta = row["pace_vs_30d_avg"] if pd.notna(row["pace_vs_30d_avg"]) else 0.0

        headline    = tmpl["headline"].format(name=row["pitcher_name"].split()[0])
        key_finding = tmpl["key_finding"].format(
            name    = row["pitcher_name"].split()[0],
            play_and_miss   = row["play_and_miss_rate"],
            dot_ball_percentage     = row["dot_ball_percentage"],
            velo    = row["avg_velo"],
            vdelta  = abs(vdelta),
            zone    = row["stump_line_rate"],
            expected_runs_per_ball   = row["avg_expected_runs_per_ball_allowed"],
            barrel  = row["boundary_rate_allowed"],
            div     = row["variation_diversity"],
        )

        insights.append({
            "pitcher_id":    int(row["pitcher_id"]),
            "pitcher_name":  row["pitcher_name"],
            "game_pk":       int(row["game_pk"]),
            "game_date":     row["game_date"].strftime("%Y-%m-%d"),
            "performance_tier": tier,
            "headline":      headline,
            "key_finding":   key_finding,
            "concern_flag":  tmpl["concern_flag"],
            "pitch_mix_note": f"Fastball-heavy mix (est. ~{random.randint(48,62)}%) with {random.choice(['slider','curveball','changeup'])} as primary off-speed." if tier in ["elite","above_avg"] else None,
        })

    return insights


def generate_players(bowlers: list, batters: list) -> pd.DataFrame:
    rows = []
    for p in bowlers:
        rows.append({"player_id": p["id"], "full_name": p["name"], "team": p["team"], "position": "SP", "hand": p["hand"]})
    for b in batters:
        rows.append({"player_id": b["id"], "full_name": b["name"], "team": b["team"], "position": "OF/1B", "hand": "R"})
    return pd.DataFrame(rows)


def _tier(dot_ball_percentage: float) -> str:
    if dot_ball_percentage >= 0.32: return "elite"
    if dot_ball_percentage >= 0.29: return "above_avg"
    if dot_ball_percentage >= 0.26: return "average"
    if dot_ball_percentage >= 0.22: return "below_avg"
    return "poor"


if __name__ == "__main__":
    print("Generating synthetic demo data...")

    pitcher_df = generate_pitcher_game_summary()
    batter_df  = generate_batter_game_summary()
    insights   = generate_llm_insights(pitcher_df)
    players_df = generate_players(BOWLERS, BATTERS)

    pitcher_df.to_parquet(OUT_DIR / "pitcher_game_summary.parquet", index=False)
    batter_df.to_parquet(OUT_DIR  / "batter_game_summary.parquet",  index=False)
    players_df.to_parquet(OUT_DIR / "players.parquet",              index=False)

    with open(OUT_DIR / "llm_insights.json", "w") as f:
        json.dump(insights, f, indent=2)

    print(f"Generated {len(pitcher_df)} bowler-game rows")
    print(f"Generated {len(batter_df)} batsman-game rows")
    print(f"Generated {len(insights)} pre-generated LLM insights")
    print(f"Generated {len(players_df)} players")
    print(f"\nFiles written to: {OUT_DIR.resolve()}")
