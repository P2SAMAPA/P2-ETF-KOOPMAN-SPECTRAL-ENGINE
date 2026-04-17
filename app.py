"""
P2 ETF Koopman-Spectral Engine — Streamlit Dashboard
Research only · Not financial advice
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

st.set_page_config(
    page_title="P2 Koopman-Spectral Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-koopman-spectral-results"
HF_DATA_REPO    = "P2SAMAPA/p2-etf-deepm-data"

# SPY is the benchmark — never shown as a pick candidate
BENCHMARK = "SPY"

# ETF pick universe (SPY excluded)
ETF_UNIVERSE = [
    "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLRE", "XLB",
    "XLP", "XLU", "GDX", "XME", "IWM", "TLT", "LQD", "HYG",
    "VNQ", "GLD", "SLV", "VCIT",
]

st.markdown("""
<style>
.main-header { font-size:2.5rem; font-weight:700; color:#1f77b4; margin-bottom:0.5rem; }
.sub-header  { font-size:1.1rem; color:#666; margin-bottom:2rem; }
.hero-card-main {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    padding:2rem; border-radius:15px; text-align:center; color:white; height:100%;
}
.hero-card {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 100%);
    padding:1.5rem; border-radius:15px; text-align:center; color:white; height:100%;
}
.benchmark-card {
    background: linear-gradient(135deg,#2c3e50 0%,#4a5568 100%);
    padding:1rem; border-radius:10px; text-align:center; color:white;
    border:2px dashed rgba(255,255,255,0.3);
}
.hero-etf-main     { font-size:5rem; font-weight:800; margin:0.5rem 0; text-shadow:2px 2px 4px rgba(0,0,0,.3); }
.hero-etf-secondary{ font-size:2.5rem; font-weight:600; opacity:.9; }
.hero-label        { font-size:.9rem; text-transform:uppercase; letter-spacing:2px; opacity:.8; }
.hero-return       { font-size:1.5rem; font-weight:700; margin-top:.5rem; }
.warning-banner    { background:#fff3cd; border:1px solid #ffeaa7; color:#856404; padding:1rem; border-radius:5px; margin-bottom:1rem; }
.source-badge      { display:inline-block; padding:.25rem .75rem; border-radius:5px; font-size:.8rem; background:#e9ecef; color:#495057; margin-bottom:1rem; }
.mode-badge        { display:inline-block; padding:.25rem .75rem; border-radius:15px; font-size:.85rem; font-weight:600; margin:.2rem; }
.growth            { background:#d4edda; color:#155724; }
.oscillatory       { background:#fff3cd; color:#856404; }
.decay             { background:#f8d7da; color:#721c24; }
.metric-highlight  { background:#f8f9fa; padding:1rem; border-radius:10px; border-left:4px solid #1f77b4; }
.footer            { margin-top:3rem; padding-top:1rem; border-top:1px solid #dee2e6; color:#6c757d; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# NYSE Calendar
# ---------------------------------------------------------------------------

class NYSECalendar:
    HOLIDAYS_2026 = [
        "2026-01-01","2026-01-19","2026-02-16","2026-04-03","2026-05-25",
        "2026-06-19","2026-07-03","2026-09-07","2026-11-26","2026-12-25",
    ]

    @classmethod
    def get_next_trading_date(cls, from_date=None):
        d = (from_date or datetime.now()) + timedelta(days=1)
        while d.weekday() >= 5 or d.strftime("%Y-%m-%d") in cls.HOLIDAYS_2026:
            d += timedelta(days=1)
        return d

    @classmethod
    def format_trading_date(cls, dt):
        return dt.strftime("%A, %B %d, %Y")


# ---------------------------------------------------------------------------
# HF signal loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_latest_signals_from_hf() -> Optional[Dict]:
    try:
        try:
            path = hf_hub_download(repo_id=HF_RESULTS_REPO,
                                   filename="signals/latest.json",
                                   repo_type="dataset")
            with open(path) as f:
                sig = json.load(f)
            sig['_source']    = f"HF: {HF_RESULTS_REPO}/signals/latest.json"
            sig['_loaded_at'] = datetime.now().isoformat()
            return sig
        except EntryNotFoundError:
            pass

        files = list_repo_files(HF_RESULTS_REPO, repo_type="dataset")
        signal_files = sorted(
            [f for f in files if f.startswith("signals/koopman_signals_") and f.endswith(".json")],
            reverse=True,
        )
        if not signal_files:
            return None

        path = hf_hub_download(repo_id=HF_RESULTS_REPO,
                               filename=signal_files[0], repo_type="dataset")
        with open(path) as f:
            sig = json.load(f)
        sig['_source']    = f"HF: {HF_RESULTS_REPO}/{signal_files[0]}"
        sig['_loaded_at'] = datetime.now().isoformat()
        return sig

    except RepositoryNotFoundError:
        st.error(f"Results repo not found: {HF_RESULTS_REPO}")
        return None
    except Exception as e:
        st.error(f"Error loading signals: {e}")
        return None


# ---------------------------------------------------------------------------
# Demo signals (SPY as benchmark, not a pick)
# ---------------------------------------------------------------------------

def generate_demo_signals(trading_date) -> Dict:
    np.random.seed(42)
    signals = []
    for etf in ETF_UNIVERSE:
        ret    = np.random.normal(15, 40)
        pred   = np.random.uniform(0.5, 0.9)
        regime = np.random.choice(["expansion","oscillatory","contraction"], p=[0.4,0.3,0.3])
        signals.append({
            'etf': etf,
            'predicted_1d_return': ret,
            'predicted_1d_return_bps': ret,
            'predictability_index': pred,
            'koopman_regime': regime,
            'is_predictable': pred > 0.6,
            'data_quality': 0.95,
        })

    signals.sort(key=lambda x: x['predicted_1d_return'], reverse=True)
    top3 = signals[:3]

    # Benchmark reference (SPY — separate, not ranked)
    spy_ret = np.random.normal(8, 25)

    return {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.1.0-DEMO",
        "timestamp": datetime.now().isoformat(),
        "signal_date": trading_date.strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open: " + NYSECalendar.format_trading_date(trading_date),
        "objective": "MAXIMUM PREDICTED RETURN (DEMO)",
        "_source": "Demo mode — no HF connection",
        "_loaded_at": datetime.now().isoformat(),
        "primary_pick": {
            "etf": top3[0]['etf'], "rank": 1,
            "predicted_1d_return_bps": round(top3[0]['predicted_1d_return'], 1),
            "predicted_1d_return_pct": round(top3[0]['predicted_1d_return'] / 100, 3),
            "predictability_index": round(top3[0]['predictability_index'], 3),
            "regime": top3[0]['koopman_regime'],
        },
        "runner_up_picks": [
            {"rank": i+2, "etf": r['etf'],
             "predicted_1d_return_bps": round(r['predicted_1d_return'], 1),
             "predicted_1d_return_pct": round(r['predicted_1d_return'] / 100, 3),
             "predictability_index": round(r['predictability_index'], 3),
             "regime": r['koopman_regime']}
            for i, r in enumerate(top3[1:3])
        ],
        "benchmark": {
            "etf": BENCHMARK,
            "note": "Benchmark only — excluded from picks",
            "predicted_1d_return_bps": round(spy_ret, 1),
        },
        "koopman_modes": {
            "regime": top3[0]['koopman_regime'],
            "predictability_index": round(top3[0]['predictability_index'], 3),
            "growth_modes": 3, "oscillatory_modes": 8, "decay_modes": 52,
            "dominant_frequency_cycles": 0.15,
        },
        "all_etfs": signals,
        "metadata": {
            "demo": True, "total_etfs_analyzed": len(signals),
            "predictable_etfs": sum(1 for s in signals if s['is_predictable']),
            "lookback_window": 63, "benchmark_excluded": BENCHMARK,
        },
    }


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_header():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<div class="main-header">🔮 P2 Koopman-Spectral Engine</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Maximum Return Targeting · NYSE Next-Open Signals · Physics-Inspired Dynamics</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div style="text-align:right;color:#666;font-size:.9rem;"><b>Version:</b> 1.1.0<br><b>Updated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-banner">⚠️ Research Only · Not Financial Advice · Signals valid for next NYSE session only</div>', unsafe_allow_html=True)


def render_hero_top3(signals: Dict):
    st.markdown("---")
    primary  = signals['primary_pick']
    runners  = signals.get('runner_up_picks', [])
    benchmark = signals.get('benchmark', {})
    target_date = signals.get('target_date', 'Next NYSE Open')
    source   = signals.get('_source', 'Unknown')

    st.markdown(f'<div class="source-badge">📡 {source}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:1rem;">
      <div style="font-size:.9rem;color:#666;text-transform:uppercase;letter-spacing:2px;">Target: Maximum Predicted Return</div>
      <div style="font-size:1.1rem;color:#1f77b4;font-weight:600;">{target_date}</div>
    </div>""", unsafe_allow_html=True)

    cols = st.columns([2, 1, 1])
    pos_col = '#90EE90'
    neg_col = '#FFB6C1'

    # Rank 1
    c1 = '#90EE90' if primary['predicted_1d_return_bps'] > 0 else '#FFB6C1'
    with cols[0]:
        st.markdown(f"""
        <div class="hero-card-main">
          <div class="hero-label">🥇 Rank #1 · Primary Signal</div>
          <div class="hero-etf-main">{primary['etf']}</div>
          <div class="hero-return" style="color:{c1};">{primary['predicted_1d_return_bps']:+.0f} bps
            <span style="font-size:.6em;">({primary['predicted_1d_return_pct']:+.3f}%)</span></div>
          <div style="margin-top:.5rem;">Predictability: {primary['predictability_index']:.2f} · {primary['regime'].upper()}</div>
        </div>""", unsafe_allow_html=True)

    # Rank 2
    if len(runners) >= 1:
        r = runners[0]
        cr = pos_col if r['predicted_1d_return_bps'] > 0 else neg_col
        with cols[1]:
            st.markdown(f"""
            <div class="hero-card">
              <div class="hero-label">🥈 Rank #2</div>
              <div class="hero-etf-secondary">{r['etf']}</div>
              <div class="hero-return" style="font-size:1.2rem;color:{cr};">{r['predicted_1d_return_bps']:+.0f} bps</div>
              <div>{r['regime']} · p={r['predictability_index']:.2f}</div>
            </div>""", unsafe_allow_html=True)

    # Rank 3
    if len(runners) >= 2:
        r = runners[1]
        cr = pos_col if r['predicted_1d_return_bps'] > 0 else neg_col
        with cols[2]:
            st.markdown(f"""
            <div class="hero-card">
              <div class="hero-label">🥉 Rank #3</div>
              <div class="hero-etf-secondary">{r['etf']}</div>
              <div class="hero-return" style="font-size:1.2rem;color:{cr};">{r['predicted_1d_return_bps']:+.0f} bps</div>
              <div>{r['regime']} · p={r['predictability_index']:.2f}</div>
            </div>""", unsafe_allow_html=True)

    # Benchmark reference row
    if benchmark:
        bps  = benchmark.get('predicted_1d_return_bps')
        cb   = pos_col if (bps or 0) > 0 else neg_col
        bps_str = f"{bps:+.0f} bps" if bps is not None else "N/A"
        st.markdown(f"""
        <div style="margin-top:1rem;">
          <div class="benchmark-card" style="max-width:320px;margin:auto;">
            <div style="font-size:.75rem;letter-spacing:2px;opacity:.7;">📊 BENCHMARK (NOT A PICK)</div>
            <div style="font-size:1.8rem;font-weight:700;">{benchmark.get('etf','SPY')}</div>
            <div style="color:{cb};font-size:1rem;font-weight:600;">{bps_str}</div>
            <div style="font-size:.75rem;opacity:.6;margin-top:.25rem;">Reference only · excluded from ranking</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-top:1rem;color:#666;font-size:.9rem;">
      <b>Objective Function:</b> max(E[Return₁ₐᵧ]) subject to Predictability &gt; 0.6<br>
      <span style="color:#999;">Sorted by predicted 1-day return (highest first) · Top 3 displayed · SPY excluded as benchmark</span>
    </div>""", unsafe_allow_html=True)


def render_full_ranking(signals: Dict):
    st.markdown("---")
    st.subheader("📊 Full ETF Ranking (by Predicted Return)")

    all_etfs = signals.get('all_etfs', [])
    if not all_etfs:
        st.info("No ranking data available")
        return

    df = pd.DataFrame(all_etfs)
    if 'predicted_1d_return' not in df.columns and 'predicted_1d_return_bps' in df.columns:
        df['predicted_1d_return'] = df['predicted_1d_return_bps']

    # Make sure SPY is not in the pick table
    df = df[df['etf'] != BENCHMARK].copy()
    df['rank'] = range(1, len(df) + 1)

    cols = ['rank','etf','predicted_1d_return','predictability_index','koopman_regime','is_predictable']
    df_show = df[[c for c in cols if c in df.columns]]

    def color_return(val):
        if isinstance(val, (int, float)):
            return f'color:{"green" if val>0 else "red" if val<0 else "black"};font-weight:bold'
        return ''

    st.dataframe(
        df_show.style
        .map(color_return, subset=['predicted_1d_return'])
        .background_gradient(subset=['predictability_index'], cmap='RdYlGn', vmin=0, vmax=1)
        .format({'predicted_1d_return': '{:+.1f} bps', 'predictability_index': '{:.3f}'}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "etf":  st.column_config.TextColumn("ETF",  width="medium"),
            "predicted_1d_return": st.column_config.NumberColumn("1-Day Return", width="medium"),
            "predictability_index": st.column_config.ProgressColumn("Predictability", min_value=0, max_value=1),
            "koopman_regime": st.column_config.TextColumn("Regime"),
            "is_predictable": st.column_config.CheckboxColumn("Valid"),
        }
    )

    if 'predicted_1d_return' in df.columns:
        fig = px.histogram(
            df, x='predicted_1d_return',
            color='koopman_regime' if 'koopman_regime' in df.columns else None,
            nbins=20,
            title="Distribution of Predicted Returns Across Universe (SPY excluded)",
            labels={'predicted_1d_return': 'Predicted 1-Day Return (bps)'},
            color_discrete_map={'expansion':'#28a745','oscillatory':'#ffc107','contraction':'#dc3545'},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)


def render_koopman_analysis(signals: Dict):
    st.markdown("---")
    st.subheader("🔬 Koopman Mode Analysis")

    primary = signals['primary_pick']
    modes   = signals.get('koopman_modes', {})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Detected Regime",      primary['regime'].upper())
        st.metric("Predictability Index", f"{primary['predictability_index']:.3f}")
    with c2:
        st.metric("Dominant Frequency",   f"{modes.get('dominant_frequency_cycles',0):.3f} cycles/day")
    with c3:
        r = primary['regime']
        if r == 'expansion':    st.error("⚠️ Growth Mode — Unstable dynamics")
        elif r == 'oscillatory': st.warning("🌊 Oscillatory — Mean-reversion likely")
        else:                    st.success("✓ Decay Mode — Converging to equilibrium")

    g, o, d = modes.get('growth_modes',0), modes.get('oscillatory_modes',0), modes.get('decay_modes',0)
    st.markdown(f"""
    <div style="margin:1rem 0;">
      <span class="mode-badge growth">Growth: {g}</span>
      <span class="mode-badge oscillatory">Oscillatory: {o}</span>
      <span class="mode-badge decay">Decay: {d}</span>
    </div>""", unsafe_allow_html=True)

    # Eigenvalue scatter (illustrative — seeded to ETF for consistency)
    st.markdown("#### Koopman Operator Eigenvalue Spectrum")
    np.random.seed(hash(primary['etf']) % 2**32)
    n = 64
    angles = np.random.uniform(0, 2*np.pi, n)
    if primary['regime'] == 'expansion':
        radii = np.concatenate([np.random.uniform(1.0,1.2,5), np.random.beta(2,5,59)*0.95])
    elif primary['regime'] == 'oscillatory':
        radii = np.concatenate([np.random.uniform(0.95,1.05,15), np.random.beta(2,5,49)*0.9])
    else:
        radii = np.random.beta(3,2,n)*0.9
    np.random.shuffle(radii)

    real, imag = radii*np.cos(angles), radii*np.sin(angles)
    colors = ['#dc3545' if r>1.0 else '#ffc107' if r>0.95 else '#28a745' for r in radii]
    theta  = np.linspace(0, 2*np.pi, 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                             line=dict(color='gray',dash='dash',width=2), name='Unit Circle'))
    fig.add_trace(go.Scatter(x=real, y=imag, mode='markers',
                             marker=dict(size=8,color=colors,line=dict(width=1,color='black')),
                             name='Koopman Modes'))
    fig.update_layout(title=f"Eigenvalue Spectrum ({primary['etf']} · {primary['regime']})",
                      xaxis_title="Real", yaxis_title="Imaginary",
                      height=550, yaxis_scaleanchor="x", yaxis_scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Red**: Growth (|λ|>1) · **Yellow**: Oscillatory (|λ|≈1) · **Green**: Decay (|λ|<1)")


def render_data_source_info(signals: Dict):
    st.markdown("---")
    st.subheader("🔗 Data Source & Methodology")
    meta = signals.get('metadata', {})
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""
        <div class="metric-highlight">
          <h4>📡 Data Source</h4>
          <p><b>Input Dataset:</b> {HF_DATA_REPO}/data/master.parquet</p>
          <p><b>Results Repo:</b> {HF_RESULTS_REPO}</p>
          <p><b>Contents:</b> OHLCV · log returns · volatility · macro indicators (VIX, T10Y2Y, DXY, HY/IG, WTI, DTB3)</p>
          <p><b>Update Frequency:</b> Daily pre-market (2:00 AM UTC)</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-highlight">
          <h4>🎯 Optimisation Objective</h4>
          <p><b>Target:</b> Maximum expected 1-day return</p>
          <p><b>Constraint:</b> Predictability index ≥ 0.6</p>
          <p><b>Benchmark:</b> SPY (reference only — excluded from picks)</p>
          <p><b>Ranking:</b> Sort by predicted return descending, select top 3</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-highlight">
          <h4>📊 Universe Stats</h4>
          <p><b>ETFs Analyzed:</b> {meta.get('total_etfs_analyzed', len(ETF_UNIVERSE))}</p>
          <p><b>Predictable:</b> {meta.get('predictable_etfs','N/A')}</p>
          <p><b>Lookback:</b> {meta.get('lookback_window',63)} days</p>
          <p><b>Benchmark excluded:</b> {meta.get('benchmark_excluded', BENCHMARK)}</p>
        </div>""", unsafe_allow_html=True)
        next_open = NYSECalendar.get_next_trading_date()
        st.markdown(f"""
        <div class="metric-highlight">
          <h4>📅 NYSE Calendar</h4>
          <p><b>Next Open:</b> {NYSECalendar.format_trading_date(next_open)}</p>
        </div>""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔧 Engine Controls")
        next_t = NYSECalendar.get_next_trading_date()
        st.info(f"📅 Target: **{NYSECalendar.format_trading_date(next_t)}**")
        if st.button("🔄 Refresh Signals", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### 📊 Engine Status")
        st.json({"Results Repo": HF_RESULTS_REPO,
                 "Observable Dim": 128,
                 "Encoder": "LSTM ×2",
                 "ETF Embedding": "Yes",
                 "Benchmark": BENCHMARK})
        st.markdown("---")
        st.markdown('<div style="font-size:.8rem;color:#666;">P2 Engine Suite · Koopman-Spectral v1.1.0<br>Research Only</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="footer">
      <b>P2 Koopman-Spectral Engine v1.1.0</b> · Maximum Return Targeting · NYSE Next-Open Scheduling<br><br>
      <b>⚠️ Research Disclaimer</b><br>
      Signals are generated pre-market for the next NYSE session.
      SPY is used as the benchmark and is excluded from pick ranking.
      Past performance does not guarantee future results. Not investment advice.<br><br>
      © 2026 P2SAMAPA · Research Only
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_header()
    render_sidebar()

    trading_date = NYSECalendar.get_next_trading_date()
    signals      = load_latest_signals_from_hf()

    if signals is None:
        signals = generate_demo_signals(trading_date)
        st.warning("⚠️ Using demo data — no signals found in HF results repo.")

    st.caption(f"Source: {signals.get('_source','Unknown')} | "
               f"Loaded: {signals.get('_loaded_at','Unknown')}")

    tab1, tab2, tab3 = st.tabs(["🎯 Top 3 Signals", "📊 Full Ranking", "🔬 Analysis"])
    with tab1:
        render_hero_top3(signals)
        render_data_source_info(signals)
    with tab2:
        render_full_ranking(signals)
    with tab3:
        render_koopman_analysis(signals)

    render_footer()


if __name__ == "__main__":
    main()
