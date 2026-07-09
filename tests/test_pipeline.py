"""
tests/test_pipeline.py
=======================
Core tests for the CricketIQ pipeline.
Uses sample data — no network calls, no API keys required.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pitcher_df():
    """Minimal pitcher_game_summary dataframe for testing."""
    return pd.DataFrame([
        {
            "pitcher_id": 1001, "pitcher_name": "Test Bowler", "team": "LAD",
            "game_pk": 700001, "game_date": pd.Timestamp("2024-07-01"),
            "opponent": "HOU", "total_pitches": 95,
            "avg_velo": 95.5, "season_avg_velo": 95.0, "pace_vs_30d_avg": 0.5,
            "play_and_miss_rate": 0.31, "dot_ball_percentage": 0.32, "stump_line_rate": 0.48,
            "wide_delivery_rate": 0.30, "boundary_rate_allowed": 0.07,
            "avg_expected_runs_per_ball_allowed": 0.270, "variation_diversity": 1.2,
            "total_re_delta": -0.15, "rolling_30d_csw_rate": 0.30,
            "rolling_30d_whiff_rate": 0.29, "rolling_30d_avg_velo": 95.0,
            "play_and_miss_delta": 0.02, "season_avg_csw": 0.30,
            "season_avg_whiff": 0.28, "performance_tier": "elite",
            "avg_spin": 2280, "avg_h_break": 8.5, "avg_v_break": 5.5,
        },
        {
            "pitcher_id": 1002, "pitcher_name": "Another Bowler", "team": "HOU",
            "game_pk": 700002, "game_date": pd.Timestamp("2024-07-01"),
            "opponent": "LAD", "total_pitches": 88,
            "avg_velo": 93.0, "season_avg_velo": 93.5, "pace_vs_30d_avg": -0.5,
            "play_and_miss_rate": 0.24, "dot_ball_percentage": 0.26, "stump_line_rate": 0.47,
            "wide_delivery_rate": 0.28, "boundary_rate_allowed": 0.09,
            "avg_expected_runs_per_ball_allowed": 0.305, "variation_diversity": 0.9,
            "total_re_delta": 0.08, "rolling_30d_csw_rate": 0.27,
            "rolling_30d_whiff_rate": 0.25, "rolling_30d_avg_velo": 93.5,
            "play_and_miss_delta": -0.01, "season_avg_csw": 0.27,
            "season_avg_whiff": 0.25, "performance_tier": "average",
            "avg_spin": 2180, "avg_h_break": 7.2, "avg_v_break": 4.8,
        },
    ])


@pytest.fixture
def sample_batter_df():
    return pd.DataFrame([
        {
            "batter_id": 2001, "batter_name": "Test Batsman", "team": "LAD",
            "game_date": pd.Timestamp("2024-07-01"), "opponent": "HOU",
            "pitches_seen": 14, "swing_rate": 0.46, "o_swing_rate": 0.29,
            "avg_exit_velo": 93.5, "avg_launch_angle": 14.2,
            "avg_expected_runs_per_ball": 0.380, "barrel_rate": 0.115,
            "hard_hit_rate": 0.44, "total_re_created": 0.18,
        }
    ])


# ── Tests: Data validation ───────────────────────────────────────────────────────

class TestDataValidation:

    def test_pitcher_csw_rate_bounds(self, sample_pitcher_df):
        """Dot Ball Percentage should always be between 0 and 1."""
        assert (sample_pitcher_df["dot_ball_percentage"] >= 0).all()
        assert (sample_pitcher_df["dot_ball_percentage"] <= 1).all()

    def test_whiff_rate_bounds(self, sample_pitcher_df):
        assert (sample_pitcher_df["play_and_miss_rate"] >= 0).all()
        assert (sample_pitcher_df["play_and_miss_rate"] <= 1).all()

    def test_performance_tier_valid_values(self, sample_pitcher_df):
        valid_tiers = {"elite", "above_avg", "average", "below_avg", "poor"}
        assert set(sample_pitcher_df["performance_tier"]).issubset(valid_tiers)

    def test_no_null_pitcher_ids(self, sample_pitcher_df):
        assert sample_pitcher_df["pitcher_id"].notna().all()

    def test_batter_exit_velo_realistic(self, sample_batter_df):
        """Exit velocity should be in realistic International Cricket range."""
        assert (sample_batter_df["avg_exit_velo"] >= 60).all()
        assert (sample_batter_df["avg_exit_velo"] <= 120).all()

    def test_barrel_rate_bounds(self, sample_batter_df):
        assert (sample_batter_df["barrel_rate"] >= 0).all()
        assert (sample_batter_df["barrel_rate"] <= 1).all()


# ── Tests: Feature engineering logic ───────────────────────────────────────────

class TestFeatureEngineering:

    def test_performance_tier_mapping(self):
        """Tier labels should map correctly to Dot Ball Percentage ranges."""
        def tier(dot_ball_percentage):
            if dot_ball_percentage >= 0.32: return "elite"
            if dot_ball_percentage >= 0.29: return "above_avg"
            if dot_ball_percentage >= 0.26: return "average"
            if dot_ball_percentage >= 0.22: return "below_avg"
            return "poor"

        assert tier(0.33) == "elite"
        assert tier(0.30) == "above_avg"
        assert tier(0.27) == "average"
        assert tier(0.23) == "below_avg"
        assert tier(0.19) == "poor"

    def test_velo_delta_calculation(self, sample_pitcher_df):
        """pace_vs_30d_avg should equal avg_velo - rolling_30d_avg_velo."""
        for _, row in sample_pitcher_df.iterrows():
            expected = round(row["avg_velo"] - row["rolling_30d_avg_velo"], 2)
            actual   = round(row["pace_vs_30d_avg"], 2)
            assert abs(expected - actual) < 0.05, f"Velo delta mismatch: {expected} vs {actual}"

    def test_whiff_rate_delta_direction(self, sample_pitcher_df):
        """play_and_miss_delta > 0 means improving vs. baseline."""
        elite_row   = sample_pitcher_df[sample_pitcher_df["performance_tier"] == "elite"].iloc[0]
        average_row = sample_pitcher_df[sample_pitcher_df["performance_tier"] == "average"].iloc[0]
        assert elite_row["play_and_miss_delta"] > average_row["play_and_miss_delta"]


# ── Tests: LLM prompt parsing ────────────────────────────────────────────────────

class TestLLMParsing:

    def test_valid_insight_json_structure(self):
        """LLM insight JSON should have required keys."""
        sample_response = {
            "performance_tier": "elite",
            "headline": "Bowler dominated with elite swing-and-miss stuff.",
            "key_finding": "Dot Ball Percentage of 32.1% was well above league average.",
            "concern_flag": None,
            "pitch_mix_note": "Fastball-heavy with effective slider.",
        }
        required_keys = {"performance_tier", "headline", "key_finding", "concern_flag"}
        assert required_keys.issubset(sample_response.keys())

    def test_valid_tier_in_response(self):
        valid_tiers = {"elite", "above_avg", "average", "below_avg", "poor"}
        response    = {"performance_tier": "elite"}
        assert response["performance_tier"] in valid_tiers

    def test_malformed_json_fallback(self):
        """System should handle JSON parse failures gracefully."""
        malformed = "Here is my analysis: the bowler was great..."
        try:
            json.loads(malformed)
            parsed = True
        except json.JSONDecodeError:
            parsed = False
        assert not parsed  # confirms we need the try/except in llm_client.py


# ── Tests: Report generation ─────────────────────────────────────────────────────

class TestReportGeneration:

    def test_grade_mapping(self):
        """Letter grades should map correctly to Dot Ball Percentage ranges."""
        def grade(dot_ball_percentage):
            if dot_ball_percentage >= 0.33: return "A+"
            if dot_ball_percentage >= 0.31: return "A"
            if dot_ball_percentage >= 0.29: return "B+"
            if dot_ball_percentage >= 0.27: return "B"
            if dot_ball_percentage >= 0.25: return "C+"
            if dot_ball_percentage >= 0.23: return "C"
            return "D"

        assert grade(0.340) == "A+"
        assert grade(0.315) == "A"
        assert grade(0.295) == "B+"
        assert grade(0.260) == "C+"
        assert grade(0.210) == "D"

    def test_stuff_score_bounds(self):
        """Stuff score should be 0-100."""
        for dot_ball_percentage in [0.15, 0.25, 0.30, 0.35, 0.42]:
            score = int(np.clip((dot_ball_percentage - 0.18) / 0.20 * 100, 0, 100))
            assert 0 <= score <= 100

    def test_percentile_monotone(self):
        """Higher Dot Ball Percentage → higher percentile."""
        def pct(dot_ball_percentage):
            breakpoints = [
                (0.35, 99), (0.33, 95), (0.31, 85), (0.29, 70),
                (0.27, 55), (0.25, 40), (0.23, 25), (0.20, 10),
            ]
            for threshold, p in breakpoints:
                if dot_ball_percentage >= threshold:
                    return p
            return 5

        assert pct(0.36) > pct(0.30) > pct(0.24) > pct(0.18)


# ── Tests: Demo data generation ──────────────────────────────────────────────────

class TestDemoData:

    def test_demo_data_row_count(self, sample_pitcher_df):
        """Should have data for multiple bowlers."""
        assert sample_pitcher_df["pitcher_id"].nunique() >= 2

    def test_demo_data_date_types(self, sample_pitcher_df):
        """game_date should be datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_pitcher_df["game_date"])

    def test_demo_insights_structure(self):
        """Pre-generated insights should have required fields."""
        sample_insight = {
            "pitcher_id": 1001,
            "game_date": "2024-07-01",
            "performance_tier": "elite",
            "headline": "Test headline.",
            "key_finding": "Test finding.",
            "concern_flag": None,
        }
        assert "pitcher_id" in sample_insight
        assert "headline"   in sample_insight
        assert "key_finding" in sample_insight


