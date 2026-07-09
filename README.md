# CricketIQ - Production International Cricket Analytics Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35-ff4b4b.svg)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-yellow.svg)](https://duckdb.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Groq API](https://img.shields.io/badge/Groq-Llama3-orange.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A portfolio-grade sports analytics system demonstrating production data engineering, machine learning, and LLM integration on International Cricket Ball Tracking data.

**[Live Demo](https://cricketiq-aovcjbmgvuzznyck63ajbg.streamlit.app/)** | **[Architecture Doc](ARCHITECTURE.md)** | **Run locally:** `streamlit run dashboard/app.py`

---

## What This Is

CricketIQ is an end-to-end International Cricket analytics platform built to demonstrate what a modern sports analytics engineering team would realistically ship. It ingests delivery-by-delivery Ball Tracking data, processes it through a Bronze/Silver/Gold medallion architecture, trains a predictive model for bowler effectiveness (Dot Ball Percentage), and surfaces results through an AI-powered scouting report system backed by Claude.

This is not a tutorial project. Every architectural decision from using DuckDB over Pandas for the analytical layer, to doing time-series cross-validation for the ML model, to positioning the LLM after all statistics are computed, reflects real production engineering judgment.

---

## Architecture

```
RAW BALL TRACKING (Cricsheet)
        |
        v
BRONZE (Parquet)       Raw, immutable, partitioned by game_date
        |
        |  cleaning · typing · dedup
        v
SILVER (DuckDB)        Normalized schema: deliveries, overs, games
                       Feature engineering via SQL window functions
        |
        |  aggregation · business logic
        v
GOLD (Parquet)         pitcher_game_summary · batter_game_summary
                       Rolling 30d averages · delta features · tiers
        |                       |
        v                       v
  XGBoost Model          LLM Enrichment
  Dot Ball Percentage prediction         Claude: narratives,
  + SHAP values          anomaly detection
        |                       |
        +-----------+-----------+
                    |
            AI Scouting Reports
            (stats + ML + LLM merged)
                    |
                    v
           Streamlit Dashboard
```

---

## Key Technical Decisions

**Why DuckDB instead of Pandas?**
DuckDB runs complex SQL window functions (30-day rolling averages, cross-bowler percentiles) on millions of rows in seconds with no cluster. The Gold layer aggregation SQL is more readable, testable, and 3-5x faster than equivalent Pandas code. It also produces Parquet-native output, keeping the pipeline stateless.

**Why Dot Ball Percentage as the ML target?**
Dot Ball Percentage (which we define in our model based on dot balls and play-and-miss) is the strongest single-game bowler quality signal, preferred by analysts over economy rate (noisy, defense-dependent) or raw wickets (does not capture command). Dot Ball Percentage is also more stable across small samples, making it a better regression target.

**Why time-series cross-validation?**
Standard k-fold shuffles the data, allowing the model to train on future games. In production, you never have future data. TimeSeriesSplit preserves temporal ordering each validation fold is strictly after its training fold. This is the most common ML leakage mistake in sports analytics portfolios.

**Why LLM as a layer, not the core?**
The LLM receives fully-computed statistics and writes narrative around them. It never computes numbers. This makes the system auditable (every stat is verifiable), testable (outputs do not change if the LLM is replaced), and cost-efficient (one API call per report, not per calculation).

---

## Project Structure

```
cricketiq/
├── data/
│   ├── bronze/           Raw Parquet, partitioned by game_date
│   ├── silver/           DuckDB analytical database
│   └── gold/             Aggregated Parquet (checked into repo for demo)
│       ├── pitcher_game_summary.parquet
│       ├── batter_game_summary.parquet
│       └── llm_insights.json
├── pipeline/
│   ├── ingestion/        statcast_ingestion.py
│   ├── silver/           cleaning.py, feature_engineering.py
│   └── gold/             aggregations.py
├── enrichment/           llm_client.py, prompt_templates.py
├── models/               train.py, evaluate.py, predict.py
├── reports/              scouting_report.py
├── dashboard/            app.py (Streamlit, 4 pages)
├── tests/
├── generate_demo_data.py
├── Makefile
├── .streamlit/           Streamlit configuration
└── pyproject.toml
```

---

## See The Project

The fastest way to explore CricketIQ is through the Streamlit dashboard. The repo includes pre-generated demo data in `data/gold/`, so visitors can run the app without API keys or a full data pipeline.

```bash
git clone https://github.com/darkphoneix/cricketiq
cd cricketiq
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Then open the local URL Streamlit prints in the terminal, usually:

```text
http://localhost:8501
```

The dashboard uses cached synthetic Ball Tracking-style data and cached LLM-style insights, so the project is viewable immediately after install.

---

## Quickstart

**Option A: Dashboard demo**

```bash
git clone https://github.com/darkphoneix/cricketiq
cd cricketiq
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The demo uses pre-generated synthetic Ball Tracking data included in `data/gold/`. No API keys needed for cached reports.

**Option B: Full pipeline with real Ball Tracking data**

```bash
cp .env.example .env
# Add GROQ_API_KEY to .env

make ingest DATE_START=2024-07-01 DATE_END=2024-07-07
make clean && make gold
make enrich DATE=2024-07-07
make train
make report PLAYER_ID=605483 DATE=2024-07-06
make dashboard
```

---

## Public Demo Deployment

This project is designed to be shared through Streamlit Community Cloud.

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Create a new app from the fork
4. Set the app file to `dashboard/app.py`
5. Deploy

No secrets are required for the demo dashboard because the repo includes cached data in `data/gold/`. Add `GROQ_API_KEY` only if you want to regenerate live LLM scouting insights.

---

## ML Model: Bowler Effectiveness

**Target:** Dot Ball Percentage

**Features:**
- `rolling_30d_csw_rate` — recent baseline
- `rolling_30d_whiff_rate` — swing-and-miss trending
- `pace_vs_30d_avg` — velocity delta vs. recent mean
- `play_and_miss_delta` — play_and_miss improvement/decline
- `variation_diversity` — delivery mix entropy
- `stump_line_rate`, `wide_delivery_rate` — command metrics
- `boundary_rate_allowed`, `avg_expected_runs_per_ball_allowed` — contact quality

**Model:** XGBoost Regressor
**Validation:** 5-fold TimeSeriesSplit (no leakage)
**Explainability:** SHAP values per prediction, surfaced in scouting reports

---

## Example Scouting Report Output

```
CRICKETIQ SCOUTING REPORT · Internal Use Only

BOWLER: Jasprit Bumrah (IND)
GAME: July 6, 2024 vs. AUS    GRADE: A (92nd percentile)

GAME STATS
  Deliveries:     98
  Avg Velo:    146.2 kph  (+0.7 vs. 30d avg)
  Play and Miss Rate:  33.8%  ELITE
  Dot Ball Percentage:    32.1%  ELITE
  expected_runs_per_ball:       .261

ML PREDICTION
  Next match projected Dot Ball Percentage:  30.8%
  Top SHAP drivers:
    1. rolling_30d_whiff_rate   +0.018
    2. pace_vs_30d_avg          +0.011
    3. variation_diversity          +0.007

AI ANALYST SUMMARY
  Tier: ELITE

  "Bumrah's yorker was historic generated 41% play_and_miss rate
  against Australia's right-handed lineup."

  Key Finding: Elite velocity held above 145 kph through the
  18th over with no fatigue-related decline. Delivery mix
  entropy suggests unpredictable sequencing that kept batsmen
  from sitting on any single delivery. The .261 expected_runs_per_ball allowed
  ranks in the 89th percentile for bowlers this season.

  Concern Flag: None identified.
```

---

## Data Source

International Cricket Ball Tracking via [Cricsheet](https://github.com/jldbc/pybaseball) - free, public, ~3M delivery events per International Cricket season. The demo ships with synthetic data calibrated to real 2024 Ball Tracking distributions. Run `make ingest` to pull live data.

---

## Stack

| Layer | Technology |
|---|---|
| Data ingestion | Cricsheet, pandas, pyarrow |
| Bronze storage | Apache Parquet (partitioned) |
| Analytical engine | DuckDB 0.10 |
| Feature engineering | DuckDB SQL window functions |
| ML model | XGBoost + SHAP |
| LLM enrichment | Groq (llama3-70b-8192) |
| Dashboard | Streamlit + Plotly |
| Task runner | GNU Make |
| Testing | pytest |
| Deployment | Streamlit Community Cloud |

---

## License

MIT
 
 