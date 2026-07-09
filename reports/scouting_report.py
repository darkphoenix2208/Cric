"""
reports/scouting_report.py
==========================
Assembles AI-powered scouting reports by merging:
    - Gold layer statistics (pitcher_game_summary)
    - ML model predictions + SHAP explanations
    - LLM-generated insights (llm_insights)

Outputs:
    - JSON (for API / dashboard consumption)
    - Console-formatted text report
    - (Optional) HTML/PDF via Jinja2 + weasyprint

Usage:
    python -m reports.scouting_report \
        --bowler-id 605483 \
        --game-date 2024-07-14
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from models.train import FEATURE_COLS, load_model
from pipeline.config import DUCKDB_PATH, REPORTS_DIR

logger = logging.getLogger(__name__)

ARTIFACTS_DIR  = Path("models/artifacts")
MODEL_PATH     = ARTIFACTS_DIR / "pitcher_effectiveness_v1.pkl"

LEAGUE_THRESHOLDS = {
    "dot_ball_percentage":            {"elite": 0.32, "above_avg": 0.29, "average": 0.26},
    "play_and_miss_rate":          {"elite": 0.33, "above_avg": 0.28, "average": 0.23},
    "avg_expected_runs_per_ball_allowed":   {"elite": 0.27, "above_avg": 0.29, "average": 0.31},  # lower is better
    "boundary_rate_allowed": {"elite": 0.05, "above_avg": 0.07, "average": 0.09},  # lower is better
}


# ─── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class PitcherGameStats:
    pitcher_id:        int
    pitcher_name:      str
    game_date:         str
    game_pk:           int
    opponent:          str
    total_pitches:     int
    avg_velo:          float
    season_avg_velo:   float
    velo_delta:        float
    play_and_miss_rate:        float
    dot_ball_percentage:          float
    stump_line_rate:         float
    wide_delivery_rate:        float
    barrel_rate:       float
    avg_expected_runs_per_ball:         float
    variation_diversity:   float
    total_re_delta:    float
    performance_tier:  str
    rolling_30d_csw:   float
    rolling_30d_whiff: float


@dataclass
class ModelPrediction:
    predicted_csw_rate: float
    stuff_score:        int     # 0-100 normalized
    top_shap_features:  list[dict[str, Any]]   # [{feature, contribution, direction}]


@dataclass
class ScoutingReport:
    # Identity
    pitcher_id:    int
    pitcher_name:  str
    game_date:     str
    opponent:      str

    # Stats
    stats:         PitcherGameStats

    # ML
    prediction:    ModelPrediction

    # LLM narrative
    headline:      str
    key_finding:   str
    concern_flag:  str | None
    pitch_mix_note: str | None

    # Ratings
    letter_grade:  str
    percentile:    int


# ─── Report Assembly ────────────────────────────────────────────────────────────

class ScoutingReportEngine:

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self.con   = con
        self.model, self.feature_cols = load_model(MODEL_PATH)

    def generate(self, pitcher_id: int, game_date: str) -> ScoutingReport:
        """Full pipeline: stats → prediction → LLM insight → report."""

        stats      = self._fetch_stats(pitcher_id, game_date)
        prediction = self._run_prediction(stats)
        llm_data   = self._fetch_llm_insight(pitcher_id, game_date)

        report = ScoutingReport(
            pitcher_id    = pitcher_id,
            pitcher_name  = stats.pitcher_name,
            game_date     = game_date,
            opponent      = stats.opponent,
            stats         = stats,
            prediction    = prediction,
            headline      = llm_data.get("headline", "No headline generated."),
            key_finding   = llm_data.get("key_finding", "Insight not available."),
            concern_flag  = llm_data.get("concern_flag"),
            pitch_mix_note= llm_data.get("pitch_mix_note"),
            letter_grade  = self._compute_grade(stats.dot_ball_percentage),
            percentile    = self._compute_percentile(stats.dot_ball_percentage),
        )

        return report

    def _fetch_stats(self, pitcher_id: int, game_date: str) -> PitcherGameStats:
        row = self.con.execute("""
            SELECT
                pgs.*,
                p.full_name,
                CASE WHEN g.home_pitcher_id = pgs.pitcher_id
                     THEN g.away_team ELSE g.home_team END AS opponent
            FROM pitcher_game_summary pgs
            LEFT JOIN players p ON p.player_id = pgs.pitcher_id
            LEFT JOIN games   g ON g.game_pk   = pgs.game_pk
            WHERE pgs.pitcher_id = ? AND pgs.game_date = ?
            LIMIT 1
        """, [pitcher_id, game_date]).df()

        if row.empty:
            raise ValueError(f"No stats found for bowler {pitcher_id} on {game_date}")

        r = row.iloc[0]
        return PitcherGameStats(
            pitcher_id        = pitcher_id,
            pitcher_name      = r.get("full_name", f"Bowler #{pitcher_id}"),
            game_date         = game_date,
            game_pk           = int(r.game_pk),
            opponent          = r.get("opponent", "Unknown"),
            total_pitches     = int(r.total_pitches),
            avg_velo          = round(float(r.avg_velo), 1),
            season_avg_velo   = round(float(r.season_avg_velo), 1),
            velo_delta        = round(float(r.pace_vs_30d_avg or 0), 1),
            play_and_miss_rate        = round(float(r.play_and_miss_rate or 0), 4),
            dot_ball_percentage          = round(float(r.dot_ball_percentage or 0), 4),
            stump_line_rate         = round(float(r.stump_line_rate or 0), 4),
            wide_delivery_rate        = round(float(r.wide_delivery_rate or 0), 4),
            barrel_rate       = round(float(r.boundary_rate_allowed or 0), 4),
            avg_expected_runs_per_ball         = round(float(r.avg_expected_runs_per_ball_allowed or 0), 4),
            variation_diversity   = round(float(r.variation_diversity or 0), 4),
            total_re_delta    = round(float(r.total_re_delta or 0), 3),
            performance_tier  = r.performance_tier,
            rolling_30d_csw   = round(float(r.rolling_30d_csw_rate or 0), 4),
            rolling_30d_whiff = round(float(r.rolling_30d_whiff_rate or 0), 4),
        )

    def _run_prediction(self, stats: PitcherGameStats) -> ModelPrediction:
        """Run XGBoost inference + SHAP attribution for one bowler-game."""

        # Build feature row (use rolling averages for forward-looking features)
        feature_row = {
            "rolling_30d_avg_velo":   stats.avg_velo,          # current = next proxy
            "rolling_30d_whiff_rate": stats.rolling_30d_whiff,
            "rolling_30d_csw_rate":   stats.rolling_30d_csw,
            "pace_vs_30d_avg":        stats.velo_delta,
            "play_and_miss_delta":       stats.play_and_miss_rate - stats.rolling_30d_whiff,
            "avg_spin":               0.0,   # would be populated in prod
            "avg_h_break":            0.0,
            "avg_v_break":            0.0,
            "variation_diversity":        stats.variation_diversity,
            "stump_line_rate":              stats.stump_line_rate,
            "wide_delivery_rate":             stats.wide_delivery_rate,
            "boundary_rate_allowed":    stats.barrel_rate,
            "avg_expected_runs_per_ball_allowed":      stats.avg_expected_runs_per_ball,
            "total_pitches":          stats.total_pitches,
            "home_away":              1,
        }

        X = pd.DataFrame([feature_row])[self.feature_cols]

        predicted_csw = float(self.model.predict(X)[0])

        # SHAP attribution
        explainer   = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)[0]

        shap_pairs = sorted(
            zip(self.feature_cols, shap_values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        top_shap = [
            {
                "feature":      feat,
                "contribution": round(float(val), 5),
                "direction":    "↑ positive" if val > 0 else "↓ negative",
            }
            for feat, val in shap_pairs[:3]
        ]

        # Stuff score: normalize predicted Dot Ball Percentage to 0-100
        # League range: ~0.18 (poor) to ~0.38 (elite)
        stuff_score = int(np.clip((predicted_csw - 0.18) / 0.20 * 100, 0, 100))

        return ModelPrediction(
            predicted_csw_rate = round(predicted_csw, 4),
            stuff_score        = stuff_score,
            top_shap_features  = top_shap,
        )

    def _fetch_llm_insight(self, pitcher_id: int, game_date: str) -> dict:
        """Pull pre-generated LLM insight from Gold layer."""
        try:
            row = self.con.execute("""
                SELECT insight_json
                FROM llm_insights
                WHERE pitcher_id = ? AND game_date = ? AND insight_type = 'pitcher_game'
                LIMIT 1
            """, [pitcher_id, game_date]).fetchone()

            if row:
                return json.loads(row[0])
        except Exception as e:
            logger.warning("Could not fetch LLM insight: %s", e)

        return {
            "headline":      "Insight generation pending.",
            "key_finding":   "Run enrichment pipeline to generate AI narrative.",
            "concern_flag":  None,
            "pitch_mix_note": None,
        }

    def _compute_grade(self, dot_ball_percentage: float) -> str:
        if dot_ball_percentage >= 0.33:  return "A+"
        if dot_ball_percentage >= 0.31:  return "A"
        if dot_ball_percentage >= 0.29:  return "B+"
        if dot_ball_percentage >= 0.27:  return "B"
        if dot_ball_percentage >= 0.25:  return "C+"
        if dot_ball_percentage >= 0.23:  return "C"
        return "D"

    def _compute_percentile(self, dot_ball_percentage: float) -> int:
        """Approximate percentile vs. league distribution (2024 baseline)."""
        breakpoints = [
            (0.35, 99), (0.33, 95), (0.31, 85), (0.29, 70),
            (0.27, 55), (0.25, 40), (0.23, 25), (0.20, 10),
        ]
        for threshold, pct in breakpoints:
            if dot_ball_percentage >= threshold:
                return pct
        return 5


# ─── Text Renderer ──────────────────────────────────────────────────────────────

def render_text_report(report: ScoutingReport) -> str:
    s  = report.stats
    p  = report.prediction
    w  = 61  # width

    lines = [
        "═" * w,
        "  BASEBALLIQ SCOUTING REPORT",
        f"  Generated: {report.game_date}  |  CONFIDENTIAL — Internal Use",
        "═" * w,
        "",
        f"BOWLER: {report.pitcher_name}",
        f"GAME:    {report.game_date} vs. {report.opponent}",
        f"GRADE:   {report.letter_grade}  ({report.percentile}th percentile)",
        "",
        "─── GAME PERFORMANCE " + "─" * 40,
        f"  Deliveries Thrown:   {s.total_pitches}",
        f"  Avg Velocity:     {s.avg_velo} mph  (season: {s.season_avg_velo}, Δ{s.velo_delta:+.1f})",
        f"  Play and Miss Rate:       {s.play_and_miss_rate:.1%}  (lg avg: 25.4%)  {'★ ELITE' if s.play_and_miss_rate >= 0.32 else ''}",
        f"  Dot Ball Percentage:         {s.dot_ball_percentage:.1%}  (lg avg: 28.1%)  {'★ ELITE' if s.dot_ball_percentage >= 0.31 else ''}",
        f"  Zone Rate:        {s.stump_line_rate:.1%}",
        f"  Chase Rate:       {s.wide_delivery_rate:.1%}",
        f"  expected_runs_per_ball Allowed:    {s.avg_expected_runs_per_ball:.3f}  (lg avg: .312)",
        f"  Barrel Rate:      {s.barrel_rate:.1%}",
        f"  Run Value:        {s.total_re_delta:+.2f} runs",
        "",
        "─── ML PREDICTION (next start) " + "─" * 30,
        f"  Predicted Dot Ball Percentage:   {p.predicted_csw_rate:.1%}",
        f"  Stuff Score:     {p.stuff_score} / 100",
        "  Top drivers:",
        *[f"    {i+1}. {f['feature']} ({f['direction']}, {f['contribution']:+.4f})"
          for i, f in enumerate(p.top_shap_features)],
        "",
        "─── AI ANALYST SUMMARY " + "─" * 38,
        f"  Tier: {s.performance_tier.upper()}",
        "",
        f"  HEADLINE: {report.headline}",
        "",
        "  KEY FINDING:",
        *[f"  {line}" for line in _wrap(report.key_finding, 56)],
        "",
    ]

    if report.concern_flag:
        lines += [
            "  ⚠ CONCERN FLAG:",
            *[f"  {line}" for line in _wrap(report.concern_flag, 56)],
            "",
        ]

    if report.pitch_mix_note:
        lines += [
            "  DELIVERY MIX NOTE:",
            *[f"  {line}" for line in _wrap(report.pitch_mix_note, 56)],
            "",
        ]

    lines.append("═" * w)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    """Simple word wrapper."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines


# ─── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate AI scouting report")
    parser.add_argument("--bowler-id", type=int, required=True)
    parser.add_argument("--game-date", required=True)
    parser.add_argument("--json-out", help="Optional path to save JSON report")
    args = parser.parse_args()

    with duckdb.connect(str(DUCKDB_PATH)) as con:
        engine = ScoutingReportEngine(con)
        report = engine.generate(args.pitcher_id, args.game_date)

        # Print formatted report
        print(render_text_report(report))

        # Optionally save JSON
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)
            logger.info("JSON report saved: %s", out)
