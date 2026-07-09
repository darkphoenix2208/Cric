"""
dashboard/app.py
================
CricketIQ — Production International Cricket Analytics Dashboard
Deployable on Streamlit Community Cloud (free tier).

Architecture:
- Loads pre-generated Parquet/JSON from data/gold/
- LLM reports generated on-demand via Groq API (st.secrets)
- No database required at runtime — pure Parquet + pandas
- ~300MB memory footprint (well within 1GB free tier limit)

Deploy:
    1. Push repo to GitHub (include data/gold/ files)
    2. Go to share.streamlit.io → New app → select repo
    3. Add GROQ_API_KEY in app Settings → Secrets
    4. Deploy ✓
"""

import json
import os
from pathlib import Path

import groq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CricketIQ",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Luxury Front Office aesthetic ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg:         #09090b;
    --bg2:        #18181b;
    --surface:    rgba(255, 255, 255, 0.03);
    --border:     rgba(0, 243, 255, 0.15);
    --border2:    rgba(255, 255, 255, 0.1);
    --accent:     #00f3ff;
    --accent2:    #7000ff;
    --glow:       rgba(0, 243, 255, 0.4);
    --text:       #f8fafc;
    --text2:      #cbd5e1;
    --muted:      #94a3b8;
    --ink:        #ffffff;
    --elite:      #39ff14;
    --above:      #00f3ff;
    --average:    #facc15;
    --below:      #f97316;
    --poor:       #ef4444;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text2) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border2) !important; }
[data-testid="stSidebar"] label { color: var(--text) !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 14px !important; }
[data-testid="stSidebar"] label p { color: var(--text) !important; font-family: 'Space Grotesk', sans-serif !important; font-size: 15px !important; font-weight: 500 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted) !important; font-family: 'Fira Code', monospace !important; font-size: 10px !important; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--accent) !important; font-family: 'Fira Code', monospace !important; font-size: 10px !important; letter-spacing: 0.15em; text-transform: uppercase; }
[data-testid="stSidebar"] .stRadio label { color: var(--text2) !important; font-family: 'Inter', sans-serif !important; font-size: 13px !important; }

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Modern header — top of every section */
.masthead {
    border-bottom: 1px solid var(--border);
    padding: 10px 0;
    margin-bottom: 24px;
    display: flex;
    align-items: baseline;
    gap: 16px;
    background: linear-gradient(90deg, rgba(0,243,255,0.05) 0%, transparent 100%);
    border-left: 4px solid var(--accent);
    padding-left: 16px;
}
.masthead-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    text-shadow: 0 0 10px var(--glow);
}
.masthead-sub {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
}
.masthead-date {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-left: auto;
    letter-spacing: 0.05em;
}

.metric-card {
    background: var(--surface);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px 16px;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(0, 243, 255, 0.15);
    border: 1px solid var(--accent);
}
.metric-label {
    font-family: 'Fira Code', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.metric-delta {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: var(--muted);
    margin-top: 8px;
}

.tier-badge {
    display: inline-block;
    padding: 4px 12px;
    font-family: 'Fira Code', monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 4px;
    border: 1px solid currentColor;
    background: rgba(0,0,0,0.5);
}
.tier-elite     { color: var(--elite); box-shadow: 0 0 8px rgba(57, 255, 20, 0.3); }
.tier-above_avg { color: var(--above); box-shadow: 0 0 8px rgba(0, 243, 255, 0.3); }
.tier-average   { color: var(--average); box-shadow: 0 0 8px rgba(250, 204, 21, 0.3); }
.tier-below_avg { color: var(--below); }
.tier-poor      { color: var(--poor); }

.section-header {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding: 8px 0;
    margin-bottom: 20px;
    margin-top: 10px;
}

/* Broadsheet column rule -> Sleek divider */
.col-rule {
    border-left: 1px solid var(--border2);
    padding-left: 20px;
    margin-left: 20px;
}

.report-block {
    background: var(--surface);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 12px;
    border-left: 4px solid var(--accent);
    padding: 24px 28px;
    font-family: 'Inter', sans-serif;
    line-height: 1.7;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.report-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 16px;
    line-height: 1.3;
    border-bottom: 1px solid var(--border2);
    padding-bottom: 12px;
}
.report-finding {
    color: var(--text2);
    font-size: 14px;
    margin-bottom: 0;
}
.report-concern {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-left: 3px solid var(--poor);
    border-radius: 6px;
    padding: 12px 16px;
    color: #fca5a5;
    font-size: 13px;
    margin-top: 16px;
    font-family: 'Fira Code', monospace;
}

.logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    text-shadow: 0 0 15px var(--glow);
}
.logo-sub {
    font-family: 'Fira Code', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
}

/* Classified stamp -> System Log Stamp */
.classified {
    display: inline-block;
    border: 1px solid var(--accent2);
    color: var(--accent2);
    background: rgba(112, 0, 255, 0.1);
    font-family: 'Fira Code', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.15em;
    padding: 4px 10px;
    border-radius: 4px;
    text-transform: uppercase;
}

/* Plotly */
.js-plotly-plot { background: transparent !important; }

/* Kill the black top bar */
[data-testid="stHeader"] {
    background-color: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stToolbar"] { background-color: var(--bg) !important; }
header[data-testid="stHeader"] * { color: var(--text2) !important; }

/* Streamlit overrides */
[data-testid="stMetric"] {
    background: var(--surface);
    backdrop-filter: blur(10px);
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: 12px;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {
    color: var(--text2) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: var(--muted) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
.stTabs [data-baseweb="tab-border"] { background-color: var(--border2) !important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }

/* Buttons */
.stButton button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    background: rgba(0, 243, 255, 0.1) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 6px !important;
    transition: all 0.2s;
}
.stButton button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 15px var(--glow) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Fira Code', monospace !important;
}

/* Sidebar radio */
div[data-testid="stSidebar"] .stRadio > div {
    gap: 4px;
}
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter", size=11),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(0,243,255,0.2)",
            tickfont=dict(size=10, color="#94a3b8"),
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(0,243,255,0.2)",
            tickfont=dict(size=10, color="#94a3b8"),
        ),
        legend=dict(font=dict(color="#cbd5e1", size=11)),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)

TIER_COLORS = {
    "elite": "#39ff14", "above_avg": "#00f3ff",
    "average": "#facc15", "below_avg": "#f97316", "poor": "#ef4444",
}

# ── Data loading ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/gold")

@st.cache_data
def load_data():
    pitcher_df = pd.read_parquet(DATA_DIR / "bowler_game_summary.parquet")
    batter_df  = pd.read_parquet(DATA_DIR / "batsman_game_summary.parquet")
    players_df = pd.read_parquet(DATA_DIR / "players.parquet")
    with open(DATA_DIR / "llm_insights.json") as f:
        insights = json.load(f)
    pitcher_df["game_date"] = pd.to_datetime(pitcher_df["game_date"])
    batter_df["game_date"]  = pd.to_datetime(batter_df["game_date"])
    return pitcher_df, batter_df, players_df, insights

pitcher_df, batter_df, players_df, insights_raw = load_data()
insights_map = {(r["pitcher_id"], r["game_date"]): r for r in insights_raw}

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text" style="border-bottom:1px solid rgba(0,243,255,0.15);padding-bottom:12px;margin-bottom:4px">C·IQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Hackathon Alpha Build</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🎯 Bowler Explorer", "💥 Batsman Explorer", "📋 Scouting Reports"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="logo-sub">Season</div>', unsafe_allow_html=True)
    min_d = pitcher_df["game_date"].min().date()
    max_d = pitcher_df["game_date"].max().date()
    date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if len(date_range) == 2:
        start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start_d, end_d = pd.Timestamp(min_d), pd.Timestamp(max_d)

    st.markdown("---")
    st.markdown('<div class="logo-sub" style="margin-bottom:8px">About</div>', unsafe_allow_html=True)
    st.caption("Synthetic International Cricket Ball Tracking data · XGBoost predictions · Claude AI insights")

# Filter data to date range
p_filt = pitcher_df[(pitcher_df["game_date"] >= start_d) & (pitcher_df["game_date"] <= end_d)]
b_filt = batter_df[ (batter_df["game_date"]  >= start_d) & (batter_df["game_date"]  <= end_d)]

# ── Helper ──────────────────────────────────────────────────────────────────────
def tier_badge(tier: str) -> str:
    return f'<span class="tier-badge tier-{tier}">{tier.replace("_"," ")}</span>'

def metric_card(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>"""

# ── Helper para aplicar template y título sin conflictos ───────────────────────
def apply_template(fig, title=None, **kwargs):
    update_kwargs = dict(**PLOT_TEMPLATE["layout"])
    update_kwargs.update(kwargs)
    fig.update_layout(**update_kwargs)
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(color="#1a1410", family="IBM Plex Mono", size=12)
            )
        )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="masthead"><span class="masthead-title">Season Intelligence</span><span class="masthead-sub">CricketIQ · Internal Use Only</span><span class="masthead-date">2024 International Cricket Season</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Key metrics · filtered date range</div>', unsafe_allow_html=True)

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(metric_card("Total Games", f"{p_filt['game_pk'].nunique():,}"), unsafe_allow_html=True)
    with col2:
        elite_pct = (p_filt["performance_tier"] == "elite").mean()
        st.markdown(metric_card("Elite Outings", f"{elite_pct:.0%}", "Dot Ball Percentage ≥ 32%"), unsafe_allow_html=True)
    with col3:
        avg_csw = p_filt["dot_ball_percentage"].mean()
        st.markdown(metric_card("Avg Dot Ball Percentage", f"{avg_csw:.1%}", "lg avg: 28.1%"), unsafe_allow_html=True)
    with col4:
        avg_expected_runs_per_ball = p_filt["avg_expected_runs_per_ball_allowed"].mean()
        st.markdown(metric_card("Avg expected_runs_per_ball Allowed", f"{avg_expected_runs_per_ball:.3f}", "lg avg: .312"), unsafe_allow_html=True)
    with col5:
        avg_velo = p_filt["avg_velo"].mean()
        st.markdown(metric_card("Avg Fastball Velo", f"{avg_velo:.1f} mph"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Dot Ball Percentage trend by bowler</div>', unsafe_allow_html=True)
        top_pitchers = p_filt.groupby("pitcher_name")["dot_ball_percentage"].mean().nlargest(6).index.tolist()
        trend_data   = p_filt[p_filt["pitcher_name"].isin(top_pitchers)]

        fig = px.line(
            trend_data.sort_values("game_date"),
            x="game_date", y="dot_ball_percentage",
            color="pitcher_name",
            labels={"dot_ball_percentage": "Dot Ball Percentage", "game_date": "", "pitcher_name": ""},
        )
        fig.add_hline(y=0.281, line_dash="dot", line_color="#6b7280",
                      annotation_text="League avg", annotation_position="bottom right")
        fig.update_traces(line_width=2)
        # FIX: usar apply_template con title_text explícito
        apply_template(fig, title="Dot Ball Percentage — top bowlers")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Performance tier distribution</div>', unsafe_allow_html=True)
        tier_counts = p_filt["performance_tier"].value_counts().reindex(
            ["elite","above_avg","average","below_avg","poor"], fill_value=0
        )
        fig2 = go.Figure(go.Bar(
            x=tier_counts.values,
            y=tier_counts.index,
            orientation="h",
            marker_color=[TIER_COLORS[t] for t in tier_counts.index],
            text=tier_counts.values,
            textposition="auto",
        ))
        # FIX: apply_template sin title (el section-header ya lo describe)
        apply_template(fig2, showlegend=False, xaxis_title="Game-starts", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Leaderboard — top bowlers by Dot Ball Percentage</div>', unsafe_allow_html=True)
    lb = (
        p_filt.groupby(["pitcher_id","pitcher_name","team"])
        .agg(
            starts    =("game_pk",   "nunique"),
            avg_csw   =("dot_ball_percentage",  "mean"),
            avg_whiff =("play_and_miss_rate","mean"),
            avg_velo  =("avg_velo",  "mean"),
            avg_expected_runs_per_ball =("avg_expected_runs_per_ball_allowed","mean"),
            best_tier =("performance_tier", lambda x: x.value_counts().index[0]),
        )
        .sort_values("avg_csw", ascending=False)
        .head(10)
        .reset_index()
    )
    lb_display = lb[["pitcher_name","team","starts","avg_csw","avg_whiff","avg_velo","avg_expected_runs_per_ball","best_tier"]].copy()
    lb_display.columns = ["Bowler","Team","GS","Dot Ball Percentage%","Whiff%","Velo","expected_runs_per_ball","Top Tier"]
    lb_display["Dot Ball Percentage%"]  = lb_display["Dot Ball Percentage%"].map("{:.1%}".format)
    lb_display["Whiff%"]= lb_display["Whiff%"].map("{:.1%}".format)
    lb_display["Velo"]  = lb_display["Velo"].map("{:.1f}".format)
    lb_display["expected_runs_per_ball"] = lb_display["expected_runs_per_ball"].map("{:.3f}".format)
    # Render as styled HTML table to avoid Streamlit dark theme override
    header_cols = lb_display.columns.tolist()
    header_html = "".join([f'<th style="background:rgba(255,255,255,0.05);color:#00f3ff;font-family:Fira Code,monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;padding:12px 16px;border-bottom:1px solid rgba(0,243,255,0.2);text-align:left">{c}</th>' for c in header_cols])
    rows_html = ""
    for i, row in lb_display.iterrows():
        bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
        cells = "".join([f'<td style="padding:10px 16px;font-family:Inter,sans-serif;font-size:12px;color:#f8fafc;border-bottom:1px solid rgba(255,255,255,0.05)">{v}</td>' for v in row])
        rows_html += f'<tr style="background:{bg}; transition: background 0.2s">{cells}</tr>'
    table_html = f'''<div style="border:1px solid rgba(0,243,255,0.15);border-radius:8px;overflow:hidden;margin-top:8px;box-shadow:0 4px 20px rgba(0,0,0,0.2);background:rgba(255,255,255,0.01)">
    <table style="width:100%;border-collapse:collapse">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table></div>'''
    st.markdown(table_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BOWLER EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Bowler Explorer":
    st.markdown("## Bowler Explorer")

    pitcher_names = sorted(p_filt["pitcher_name"].unique())
    selected = st.selectbox("Select bowler", pitcher_names)
    p_data   = p_filt[p_filt["pitcher_name"] == selected].sort_values("game_date")

    if p_data.empty:
        st.warning("No data for this bowler in the selected date range.")
        st.stop()

    latest = p_data.iloc[-1]

    # Header metrics
    st.markdown(f"### {selected} — {latest['team']}")
    st.markdown(tier_badge(latest["performance_tier"]) + f" &nbsp; Last start: {latest['game_date'].strftime('%b %d, %Y')} vs. {latest['opponent']}", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    stats = [
        ("GS (filtered)", str(len(p_data))),
        ("Avg Dot Ball Percentage%",  f"{p_data['dot_ball_percentage'].mean():.1%}"),
        ("Avg Whiff%",f"{p_data['play_and_miss_rate'].mean():.1%}"),
        ("Avg Velo",  f"{p_data['avg_velo'].mean():.1f}"),
        ("expected_runs_per_ball",     f"{p_data['avg_expected_runs_per_ball_allowed'].mean():.3f}"),
        ("RE Delta",  f"{p_data['total_re_delta'].sum():+.1f}"),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4,c5,c6], stats):
        with col:
            st.markdown(metric_card(label, val), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📈 Performance Trends", "🎯 Command & Stuff", "📊 Season Summary"])

    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=p_data["game_date"], y=p_data["dot_ball_percentage"],
                mode="lines+markers", name="Dot Ball Percentage",
                line=dict(color="#e8c84a", width=2),
                marker=dict(size=6, color=[TIER_COLORS[t] for t in p_data["performance_tier"]]),
            ))
            fig.add_hline(y=0.281, line_dash="dot", line_color="#6b7280",
                          annotation_text="Lg avg", annotation_position="bottom right")
            # FIX: title_text explícito con apply_template
            apply_template(fig, title="Dot Ball Percentage — game by game")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=p_data["game_date"], y=p_data["avg_velo"],
                mode="lines+markers", name="Game Velo",
                line=dict(color="#4a9eff", width=2),
            ))
            fig2.add_trace(go.Scatter(
                x=p_data["game_date"], y=p_data["rolling_30d_avg_velo"],
                mode="lines", name="30d Rolling Avg",
                line=dict(color="#6b7280", width=1.5, dash="dot"),
            ))
            # FIX: title_text explícito con apply_template
            apply_template(fig2, title="Velocity Trend")
            fig2.update_yaxes(title_text="mph")
            st.plotly_chart(fig2, use_container_width=True)

        # Whiff rate with rolling baseline
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=p_data["game_date"], y=p_data["play_and_miss_rate"],
            name="Play and Miss Rate",
            marker_color=[TIER_COLORS[t] for t in p_data["performance_tier"]],
            opacity=0.8,
        ))
        fig3.add_trace(go.Scatter(
            x=p_data["game_date"], y=p_data["rolling_30d_whiff_rate"],
            mode="lines", name="30d Avg",
            line=dict(color="#e8c84a", width=2, dash="dot"),
        ))
        fig3.add_hline(y=0.254, line_dash="dash", line_color="#6b7280",
                       annotation_text="Lg avg 25.4%")
        # FIX: title_text explícito con apply_template
        apply_template(fig3, title="Play and Miss Rate — bars colored by performance tier")
        fig3.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        col_l, col_r = st.columns(2)
        with col_l:
            # Zone rate vs chase rate scatter
            # FIX: NO poner title en el constructor px — dejarlo en apply_template
            fig4 = px.scatter(
                p_data, x="stump_line_rate", y="wide_delivery_rate",
                color="dot_ball_percentage",
                color_continuous_scale=[(0,"#ef4444"),(0.5,"#eab308"),(1,"#22c55e")],
                size="total_pitches",
                hover_data=["game_date","opponent","dot_ball_percentage"],
                labels={"stump_line_rate":"Zone Rate","wide_delivery_rate":"Chase Rate"},
            )
            apply_template(fig4, title="Zone Rate vs Chase Rate")
            fig4.update_xaxes(tickformat=".0%")
            fig4.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig4, use_container_width=True)

        with col_r:
            # Dot Ball Percentage vs expected_runs_per_ball
            # FIX: NO poner title en el constructor px — dejarlo en apply_template
            fig5 = px.scatter(
                p_data, x="dot_ball_percentage", y="avg_expected_runs_per_ball_allowed",
                color="performance_tier",
                color_discrete_map=TIER_COLORS,
                hover_data=["game_date","opponent"],
                labels={"dot_ball_percentage":"Dot Ball Percentage","avg_expected_runs_per_ball_allowed":"expected_runs_per_ball Allowed"},
            )
            apply_template(fig5, title="Stuff Quality vs Contact Quality")
            fig5.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        summary = p_data.agg({
            "dot_ball_percentage": ["mean","max","min","std"],
            "play_and_miss_rate": ["mean","max"],
            "avg_velo": ["mean","max","min"],
            "boundary_rate_allowed": "mean",
            "avg_expected_runs_per_ball_allowed": "mean",
            "total_re_delta": "sum",
        }).round(4)

        # Formatear valores para mejor legibilidad
        summary_display = summary.copy()
        pct_cols = ["dot_ball_percentage", "play_and_miss_rate", "boundary_rate_allowed"]
        for col in summary_display.columns:
            if col in pct_cols:
                summary_display[col] = summary_display[col].apply(
                    lambda v: f"{v:.1%}" if pd.notna(v) else "—"
                )
            elif col == "avg_expected_runs_per_ball_allowed":
                summary_display[col] = summary_display[col].apply(
                    lambda v: f"{v:.3f}" if pd.notna(v) else "—"
                )
            elif col == "avg_velo":
                summary_display[col] = summary_display[col].apply(
                    lambda v: f"{v:.1f}" if pd.notna(v) else "—"
                )
            elif col == "total_re_delta":
                summary_display[col] = summary_display[col].apply(
                    lambda v: f"{v:+.3f}" if pd.notna(v) else "—"
                )
            else:
                summary_display[col] = summary_display[col].apply(
                    lambda v: str(v) if pd.notna(v) else "—"
                )

        # Renombrar columnas para display
        col_rename = {
            "dot_ball_percentage": "Dot Ball Percentage%",
            "play_and_miss_rate": "Whiff%",
            "avg_velo": "Velo",
            "boundary_rate_allowed": "Barrel%",
            "avg_expected_runs_per_ball_allowed": "expected_runs_per_ball",
            "total_re_delta": "RE Delta",
        }
        summary_display.columns = [col_rename.get(c, c) for c in summary_display.columns]

        # Render HTML igual que el Leaderboard
        s_header_cols = ["Stat"] + summary_display.columns.tolist()
        s_header_html = "".join([
            f'<th style="background:rgba(255,255,255,0.05);color:#00f3ff;font-family:Fira Code,monospace;font-size:10px;letter-spacing:0.12em;text-transform:uppercase;padding:12px 16px;border-bottom:1px solid rgba(0,243,255,0.2);text-align:left">{c}</th>'
            for c in s_header_cols
        ])
        s_rows_html = ""
        for i, (idx, row_s) in enumerate(summary_display.iterrows()):
            bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
            row_label = f'<td style="padding:10px 16px;font-family:Fira Code,monospace;font-size:11px;font-weight:500;color:#00f3ff;border-bottom:1px solid rgba(255,255,255,0.05);letter-spacing:0.08em;text-transform:uppercase">{idx}</td>'
            cells = "".join([
                f'<td style="padding:10px 16px;font-family:Inter,sans-serif;font-size:12px;color:#f8fafc;border-bottom:1px solid rgba(255,255,255,0.05)">{v}</td>'
                for v in row_s
            ])
            s_rows_html += f'<tr style="background:{bg}; transition: background 0.2s">{row_label}{cells}</tr>'

        s_table_html = f'''<div style="border:1px solid rgba(0,243,255,0.15);border-radius:8px;overflow:hidden;margin-top:8px;margin-bottom:24px;box-shadow:0 4px 20px rgba(0,0,0,0.2);background:rgba(255,255,255,0.01)">
        <table style="width:100%;border-collapse:collapse">
            <thead><tr>{s_header_html}</tr></thead>
            <tbody>{s_rows_html}</tbody>
        </table></div>'''
        st.markdown(s_table_html, unsafe_allow_html=True)

        tier_dist = p_data["performance_tier"].value_counts()
        fig6 = go.Figure(go.Pie(
            labels=tier_dist.index,
            values=tier_dist.values,
            marker_colors=[TIER_COLORS.get(t, "#666") for t in tier_dist.index],
            hole=0.45,
        ))
        # FIX: title_text explícito con apply_template
        apply_template(fig6, title="Outing distribution by tier", showlegend=True)
        st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATSMAN EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💥 Batsman Explorer":
    st.markdown("## Batsman Explorer")

    batter_names = sorted(b_filt["batter_name"].unique())
    selected_b   = st.selectbox("Select batsman", batter_names)
    b_data       = b_filt[b_filt["batter_name"] == selected_b].sort_values("game_date")

    if b_data.empty:
        st.warning("No data in selected range.")
        st.stop()

    latest_b = b_data.iloc[-1]
    st.markdown(f"### {selected_b} — {latest_b['team']}")
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    bstats = [
        ("Avg Exit Velo",  f"{b_data['avg_exit_velo'].mean():.1f} mph"),
        ("Avg expected_runs_per_ball",      f"{b_data['avg_expected_runs_per_ball'].mean():.3f}"),
        ("Barrel Rate",    f"{b_data['barrel_rate'].mean():.1%}"),
        ("Hard Hit%",      f"{b_data['hard_hit_rate'].mean():.1%}"),
        ("O-Swing%",       f"{b_data['o_swing_rate'].mean():.1%}"),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4,c5], bstats):
        with col:
            st.markdown(metric_card(label, val), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Rolling expected_runs_per_ball
        b_data = b_data.copy()
        b_data["xwoba_7d"] = b_data["avg_expected_runs_per_ball"].rolling(7, min_periods=2).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=b_data["game_date"], y=b_data["avg_expected_runs_per_ball"],
            name="Game expected_runs_per_ball", marker_color="#4a9eff", opacity=0.5,
        ))
        fig.add_trace(go.Scatter(
            x=b_data["game_date"], y=b_data["xwoba_7d"],
            mode="lines", name="7-game avg",
            line=dict(color="#e8c84a", width=2),
        ))
        fig.add_hline(y=0.312, line_dash="dot", line_color="#6b7280",
                      annotation_text="Lg avg")
        # FIX: title_text explícito con apply_template
        apply_template(fig, title="expected_runs_per_ball — game log + 7-game rolling avg")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Exit velocity distribution
        # FIX: NO poner title en el constructor px — dejarlo en apply_template
        fig2 = px.histogram(
            b_data, x="avg_exit_velo", nbins=25,
            color_discrete_sequence=["#e8c84a"],
            labels={"avg_exit_velo": "Exit Velocity (mph)"},
        )
        fig2.add_vline(x=95, line_dash="dot", line_color="#22c55e",
                       annotation_text="Hard hit threshold (95)")
        apply_template(fig2, title="Exit Velocity Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    # Barrel rate trend
    b_data["barrel_7d"] = b_data["barrel_rate"].rolling(7, min_periods=2).mean()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=b_data["game_date"], y=b_data["barrel_rate"],
        mode="markers", name="Barrel Rate",
        marker=dict(color="#f97316", size=5, opacity=0.6),
    ))
    fig3.add_trace(go.Scatter(
        x=b_data["game_date"], y=b_data["barrel_7d"],
        mode="lines", name="7-game trend",
        line=dict(color="#e8c84a", width=2),
    ))
    fig3.add_hline(y=0.078, line_dash="dot", line_color="#6b7280",
                   annotation_text="Lg avg 7.8%")
    # FIX: title_text explícito con apply_template
    apply_template(fig3, title="Barrel Rate — season arc")
    fig3.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SCOUTING REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Scouting Reports":
    st.markdown("## AI Scouting Reports")
    st.markdown('<div class="section-header">Powered by Claude · LLM insights generated on demand</div>', unsafe_allow_html=True)

    col_sel, col_date = st.columns([2, 2])
    with col_sel:
        pitcher_names = sorted(p_filt["pitcher_name"].unique())
        sel_pitcher   = st.selectbox("Bowler", pitcher_names)
    with col_date:
        p_games = pitcher_df[pitcher_df["pitcher_name"] == sel_pitcher]["game_date"].dt.date.unique()
        p_games = sorted(p_games, reverse=True)
        sel_date = st.selectbox("Game date", p_games, format_func=lambda d: d.strftime("%B %d, %Y"))

    row_data = pitcher_df[
        (pitcher_df["pitcher_name"] == sel_pitcher) &
        (pitcher_df["game_date"].dt.date == sel_date)
    ]

    if row_data.empty:
        st.warning("No data for this selection.")
        st.stop()

    row = row_data.iloc[0]

    # ── Stats display ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {sel_pitcher} · {sel_date.strftime('%b %d, %Y')} vs. {row['opponent']}")
    st.markdown(tier_badge(row["performance_tier"]), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    report_stats = [
        ("Deliveries", str(int(row["total_pitches"]))),
        ("Dot Ball Percentage", f"{row['dot_ball_percentage']:.1%}"),
        ("Play and Miss Rate", f"{row['play_and_miss_rate']:.1%}"),
        ("Avg Velo", f"{row['avg_velo']:.1f} mph"),
    ]
    for col, (label, val) in zip([c1,c2,c3,c4], report_stats):
        with col:
            st.markdown(metric_card(label, val), unsafe_allow_html=True)

    c5,c6,c7,c8 = st.columns(4)
    report_stats2 = [
        ("Zone Rate",  f"{row['stump_line_rate']:.1%}"),
        ("Chase Rate", f"{row['wide_delivery_rate']:.1%}"),
        ("expected_runs_per_ball",      f"{row['avg_expected_runs_per_ball_allowed']:.3f}"),
        ("RE Delta",   f"{row['total_re_delta']:+.2f}"),
    ]
    for col, (label, val) in zip([c5,c6,c7,c8], report_stats2):
        with col:
            st.markdown(metric_card(label, val), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── LLM Report ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">AI Analyst Report</div>', unsafe_allow_html=True)

    date_key   = row["game_date"].strftime("%Y-%m-%d")
    cached_key = (int(row["pitcher_id"]), date_key)
    cached     = insights_map.get(cached_key)

    tab_cached, tab_live = st.tabs(["📁 Cached Report", "⚡ Generate Live (API)"])

    with tab_cached:
        if cached:
            st.markdown(f"""
            <div class="report-block">
                <div class="report-headline">"{cached['headline']}"</div>
                <div class="report-finding">{cached['key_finding']}</div>
            </div>""", unsafe_allow_html=True)
            if cached.get("pitch_mix_note"):
                st.markdown(f'<p style="color:#9ca3af;font-size:13px;margin-top:4px;font-style:italic">🎯 {cached["pitch_mix_note"]}</p>', unsafe_allow_html=True)
            if cached.get("concern_flag"):
                st.markdown(f'<div class="report-concern">⚠ {cached["concern_flag"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No cached report for this game. Generate a live report using the API tab.")

    with tab_live:
        st.caption("Calls Groq API in real time. Add GROQ_API_KEY to Streamlit secrets to enable.")

        if st.button("⚡ Generate AI Report", type="primary"):
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                api_key = os.environ.get("GROQ_API_KEY", "")

            if not api_key:
                st.error("No API key found. Add GROQ_API_KEY to .streamlit/secrets.toml or Streamlit Cloud secrets.")
            else:
                with st.spinner("Calling Groq..."):
                    prompt = f"""You are a cricket analyst for an International Cricket front office.

Bowler: {sel_pitcher}
Date: {date_key}
Opponent: {row['opponent']}

Stats:
- Deliveries: {int(row['total_pitches'])}
- Avg velocity: {row['avg_velo']:.1f} mph (season avg: {row['season_avg_velo']:.1f} mph)
- Whiff rate: {row['play_and_miss_rate']:.1%} (league avg: 25.4%)
- Dot Ball Percentage: {row['dot_ball_percentage']:.1%} (league avg: 28.1%)
- Zone rate: {row['stump_line_rate']:.1%}
- Chase rate: {row['wide_delivery_rate']:.1%}
- expected_runs_per_ball allowed: {row['avg_expected_runs_per_ball_allowed']:.3f}
- Barrel rate: {row['boundary_rate_allowed']:.1%}
- Run value delta: {row['total_re_delta']:+.2f}

Performance tier: {row['performance_tier']}

Respond ONLY with valid JSON (no markdown):
{{"headline": "<15 words max>", "key_finding": "<2-3 analytical sentences>", "concern_flag": null or "<one sentence>", "pitch_mix_note": null or "<one sentence>"}}"""

                    try:
                        client   = groq.Groq(api_key=api_key)
                        response = client.chat.completions.create(
                            model="llama3-70b-8192",
                            max_tokens=600,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        result = json.loads(response.choices[0].message.content.strip())

                        st.markdown(f"""
                        <div class="report-block">
                            <div class="report-headline">"{result['headline']}"</div>
                            <div class="report-finding">{result['key_finding']}</div>
                        </div>""", unsafe_allow_html=True)
                        if result.get("pitch_mix_note"):
                            st.markdown(f'<p style="color:#9ca3af;font-size:13px;margin-top:4px;font-style:italic">🎯 {result["pitch_mix_note"]}</p>', unsafe_allow_html=True)
                        if result.get("concern_flag"):
                            st.markdown(f'<div class="report-concern">⚠ {result["concern_flag"]}</div>', unsafe_allow_html=True)

                    except json.JSONDecodeError:
                        st.error("Could not parse LLM response as JSON.")
                    except Exception as e:
                        st.error(f"API error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Mini trend chart in report view ─────────────────────────────────────
    st.markdown('<div class="section-header">Recent form — last 8 starts</div>', unsafe_allow_html=True)
    recent_starts = (
        pitcher_df[pitcher_df["pitcher_name"] == sel_pitcher]
        .sort_values("game_date")
        .tail(8)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=recent_starts["game_date"], y=recent_starts["dot_ball_percentage"],
        marker_color=[TIER_COLORS[t] for t in recent_starts["performance_tier"]],
        name="Dot Ball Percentage",
        text=[f"{v:.1%}" for v in recent_starts["dot_ball_percentage"]],
        textposition="outside",
        textfont=dict(color="#e8eaf0", size=11),
    ))
    fig.add_hline(y=0.281, line_dash="dot", line_color="#6b7280",
                  annotation_text="Lg avg")
    # Highlight selected game
    selected_ts = pd.Timestamp(sel_date)
    if selected_ts in recent_starts["game_date"].values:
        fig.add_vline(x=selected_ts.timestamp() * 1000, line_color="#e8c84a",
                      annotation_text="Selected", line_width=2)
    # FIX: apply_template sin title (el section-header ya lo describe)
    apply_template(fig, showlegend=False, height=280)
    fig.update_yaxes(tickformat=".0%", range=[0, recent_starts["dot_ball_percentage"].max() * 1.25])
    st.plotly_chart(fig, use_container_width=True)

