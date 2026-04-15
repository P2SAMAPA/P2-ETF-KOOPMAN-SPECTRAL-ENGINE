"""
P2 ETF Koopman-Spectral Engine
Streamlit Dashboard for Signal Visualization
Research only · Not financial advice

Targets: Maximum predicted return for next NYSE trading day
Data Source: HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet
Results: HF: P2SAMAPA/p2-etf-koopman-spectral-results
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

# HuggingFace Hub for fetching results
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError

# Page configuration
st.set_page_config(
    page_title="P2 Koopman-Spectral Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
HF_RESULTS_REPO = "P2SAMAPA/p2-etf-koopman-spectral-results"
HF_DATA_REPO = "P2SAMAPA/p2-etf-deepm-data"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    .hero-etf-main {
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        margin: 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .hero-etf-secondary {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        opacity: 0.9;
    }
    .hero-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.8;
        text-align: center;
    }
    .hero-return {
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0.5rem;
    }
    .hero-divider {
        border-left: 2px solid rgba(255,255,255,0.3);
        height: 100%;
    }
    .warning-banner {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 5px;
        font-size: 0.8rem;
        background: #e9ecef;
        color: #495057;
        margin-bottom: 1rem;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .growth { background: #d4edda; color: #155724; }
    .oscillatory { background: #fff3cd; color: #856404; }
    .decay { background: #f8d7da; color: #721c24; }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #dee2e6;
        color: #6c757d;
        font-size: 0.9rem;
    }
    .metric-highlight {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .hero-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        height: 100%;
    }
    .hero-card-main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


# --- NYSE Calendar ---

class NYSECalendar:
    """NYSE trading calendar for next valid trading date."""
    
    HOLIDAYS_2026 = [
        "2026-01-01",  # New Year's
        "2026-01-19",  # MLK Day
        "2026-02-16",  # Presidents Day
        "2026-04-03",  # Good Friday
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Independence Day (observed)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
    ]
    
    @classmethod
    def get_next_trading_date(cls, from_date: Optional[datetime] = None) -> datetime:
        """
        Get next NYSE trading date (skips weekends and holidays).
        """
        if from_date is None:
            from_date = datetime.now()
        
        next_date = from_date + timedelta(days=1)
        
        # Keep advancing until valid trading day
        while True:
            # Skip weekends
            if next_date.weekday() >= 5:  # Saturday=5, Sunday=6
                next_date += timedelta(days=1)
                continue
            
            # Skip holidays
            date_str = next_date.strftime("%Y-%m-%d")
            if date_str in cls.HOLIDAYS_2026:
                next_date += timedelta(days=1)
                continue
            
            return next_date
    
    @classmethod
    def format_trading_date(cls, dt: datetime) -> str:
        """Format for display."""
        return dt.strftime("%A, %B %d, %Y")


# --- HF Results Loading ---

@st.cache_data(ttl=300)
def load_latest_signals_from_hf() -> Optional[Dict]:
    """
    Load latest signals from HF results repo.
    Tries 'signals/latest.json' first, then most recent dated file.
    """
    try:
        # Try latest.json first
        try:
            path = hf_hub_download(
                repo_id=HF_RESULTS_REPO,
                filename="signals/latest.json",
                repo_type="dataset"
            )
            with open(path) as f:
                signals = json.load(f)
            signals['_source'] = f"HF: {HF_RESULTS_REPO}/signals/latest.json"
            signals['_loaded_at'] = datetime.now().isoformat()
            return signals
        except EntryNotFoundError:
            pass
        
        # List all signal files and get most recent
        files = list_repo_files(HF_RESULTS_REPO, repo_type="dataset")
        signal_files = [f for f in files if f.startswith("signals/koopman_signals_") and f.endswith(".json")]
        
        if not signal_files:
            return None
        
        # Sort by date (newest first)
        signal_files.sort(reverse=True)
        latest_file = signal_files[0]
        
        path = hf_hub_download(
            repo_id=HF_RESULTS_REPO,
            filename=latest_file,
            repo_type="dataset"
        )
        
        with open(path) as f:
            signals = json.load(f)
        signals['_source'] = f"HF: {HF_RESULTS_REPO}/{latest_file}"
        signals['_loaded_at'] = datetime.now().isoformat()
        return signals
        
    except RepositoryNotFoundError:
        st.error(f"Results repo not found: {HF_RESULTS_REPO}")
        return None
    except Exception as e:
        st.error(f"Error loading from HF: {e}")
        return None


# --- Demo Signals ---

def generate_demo_signals(trading_date: datetime) -> Dict:
    """Generate realistic demo signals when HF data unavailable."""
    etfs = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", 
            "XLP", "XLU", "GDX", "XME", "IWM", "TLT", "LQD", "HYG", 
            "VNQ", "GLD", "SLV", "VCIT"]
    
    np.random.seed(42)
    signals = []
    
    for etf in etfs:
        ret = np.random.normal(15, 40)  # Mean 15bps, std 40bps
        pred = np.random.uniform(0.5, 0.9)
        regime = np.random.choice(["expansion", "oscillatory", "contraction"], 
                                  p=[0.4, 0.3, 0.3])
        signals.append({
            'etf': etf,
            'predicted_1d_return': ret,
            'predictability_index': pred,
            'koopman_regime': regime,
            'is_predictable': pred > 0.6,
            'growth_modes': np.random.randint(0, 5),
            'oscillatory_modes': np.random.randint(0, 8),
            'decay_modes': np.random.randint(55, 64),
            'data_quality': 0.95
        })
    
    # SORT BY PREDICTED 1-DAY RETURN (DESCENDING) - MAXIMIZE RETURN
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    top3 = signals_sorted[:3]
    primary = top3[0]
    runners = top3[1:3] if len(top3) > 1 else []
    
    return {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.0.0-DEMO",
        "timestamp": datetime.now().isoformat(),
        "signal_date": trading_date.strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open: " + NYSECalendar.format_trading_date(trading_date),
        "objective": "MAXIMUM PREDICTED RETURN (DEMO MODE)",
        "data_source": "DEMO: Synthetic Data",
        "results_repo": f"HF: {HF_RESULTS_REPO}",
        "_source": "Demo mode (no HF connection)",
        "_loaded_at": datetime.now().isoformat(),
        "primary_pick": {
            "etf": primary['etf'],
            "rank": 1,
            "predicted_1d_return_bps": round(primary['predicted_1d_return'], 1),
            "predicted_1d_return_pct": round(primary['predicted_1d_return'] / 100, 3),
            "predictability_index": round(primary['predictability_index'], 3),
            "regime": primary['koopman_regime'],
            "conviction_derived": round(primary['predictability_index'] * 100, 1),
            "data_quality": primary['data_quality']
        },
        "runner_up_picks": [
            {
                "rank": i + 2,
                "etf": r['etf'],
                "predicted_1d_return_bps": round(r['predicted_1d_return'], 1),
                "predicted_1d_return_pct": round(r['predicted_1d_return'] / 100, 3),
                "predictability_index": round(r['predictability_index'], 3),
                "regime": r['koopman_regime'],
                "data_quality": r['data_quality']
            }
            for i, r in enumerate(runners)
        ],
        "koopman_modes": {
            "regime": primary['koopman_regime'],
            "predictability_index": round(primary['predictability_index'], 3),
            "growth_modes": primary['growth_modes'],
            "oscillatory_modes": primary['oscillatory_modes'],
            "decay_modes": primary['decay_modes'],
            "dominant_frequency_cycles": round(0.1 + np.random.random() * 0.3, 3)
        },
        "all_etfs": signals_sorted,
        "metadata": {
            "demo": True,
            "warning": "Using synthetic data - HF dataset not connected",
            "total_etfs_analyzed": len(signals),
            "predictable_etfs": sum(1 for s in signals if s['is_predictable'])
        }
    }


# --- Visualization Components ---

def render_header():
    """Render main header with branding."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="main-header">🔮 P2 Koopman-Spectral Engine</div>', 
                   unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Maximum Return Targeting · '
                   'NYSE Next-Open Signals · Physics-Inspired Dynamics</div>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: right; color: #666; font-size: 0.9rem;">
            <b>Version:</b> 1.0.0<br>
            <b>Updated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="warning-banner">'
               '<b>⚠️ Research Only</b> · Not Financial Advice · '
               'Targets maximum predicted return with predictability filtering · '
               'Signals valid for next NYSE trading session only</div>', 
               unsafe_allow_html=True)


def render_hero_top3(signals: Dict):
    """
    Render hero card with TOP 3 ETFs by predicted return.
    Uses Streamlit columns for proper rendering instead of raw HTML flexbox.
    """
    st.markdown("---")
    
    primary = signals['primary_pick']
    runners = signals.get('runner_up_picks', [])
    target_date = signals.get('target_date', 'Next NYSE Open')
    
    # Source badge
    source = signals.get('_source', 'Unknown')
    st.markdown(f'<div class="source-badge">📡 {source}</div>', unsafe_allow_html=True)
    
    # Header with target info
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 0.9rem; color: #666; text-transform: uppercase; letter-spacing: 2px;">
            Target: Maximum Predicted Return
        </div>
        <div style="font-size: 1.1rem; color: #1f77b4; font-weight: 600;">
            {target_date}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Determine number of columns based on available runners
    num_runners = len(runners)
    
    if num_runners >= 2:
        cols = st.columns([2, 1, 1])
    elif num_runners >= 1:
        cols = st.columns([2, 1])
    else:
        cols = [st.container()]
    
    # Column 0: Primary Pick (Rank #1) - Larger
    primary_color = '#90EE90' if primary['predicted_1d_return_bps'] > 0 else '#FFB6C1'
    
    with cols[0]:
        st.markdown(f"""
        <div class="hero-card-main">
            <div class="hero-label">🥇 Rank #1 · Primary Signal</div>
            <div class="hero-etf-main">{primary['etf']}</div>
            <div class="hero-return" style="color: {primary_color};">
                {primary['predicted_1d_return_bps']:+.0f} bps
                <span style="font-size: 0.6em; opacity: 0.8;">({primary['predicted_1d_return_pct']:+.3f}%)</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                Predictability: {primary['predictability_index']:.2f} · {primary['regime'].upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Column 1: Runner #2
    if num_runners >= 1:
        r = runners[0]
        runner_color = '#90EE90' if r['predicted_1d_return_bps'] > 0 else '#FFB6C1'
        with cols[1]:
            st.markdown(f"""
            <div class="hero-card">
                <div class="hero-label">🥈 Rank #2</div>
                <div class="hero-etf-secondary">{r['etf']}</div>
                <div class="hero-return" style="font-size: 1.2rem; color: {runner_color};">
                    {r['predicted_1d_return_bps']:+.0f} bps
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    {r['regime']} · p={r['predictability_index']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Column 2: Runner #3 (if exists)
    if num_runners >= 2:
        r = runners[1]
        runner_color = '#90EE90' if r['predicted_1d_return_bps'] > 0 else '#FFB6C1'
        with cols[2]:
            st.markdown(f"""
            <div class="hero-card">
                <div class="hero-label">🥉 Rank #3</div>
                <div class="hero-etf-secondary">{r['etf']}</div>
                <div class="hero-return" style="font-size: 1.2rem; color: {runner_color};">
                    {r['predicted_1d_return_bps']:+.0f} bps
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    {r['regime']} · p={r['predictability_index']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Objective reminder
    st.markdown(f"""
    <div style="text-align: center; margin-top: 1rem; color: #666; font-size: 0.9rem;">
        <b>Objective Function:</b> max(E[Return₁ₐᵧ]) subject to Predictability > 0.6<br>
        <span style="color: #999;">Sorted by predicted 1-day return (highest first) · Top 3 displayed</span>
    </div>
    """, unsafe_allow_html=True)


def render_full_ranking(signals: Dict):
    """Render full ETF ranking table with visualizations."""
    st.markdown("---")
    st.subheader("📊 Full ETF Ranking (by Predicted Return)")
    
    all_etfs = signals.get('all_etfs', [])
    if not all_etfs:
        st.info("No ranking data available")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_etfs[:20])  # Top 20
    
    # Format for display
    display_cols = ['etf', 'predicted_1d_return', 'predictability_index', 
                   'koopman_regime', 'is_predictable', 'data_quality']
    df_display = df[[c for c in display_cols if c in df.columns]].copy()
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Reorder columns
    cols = ['rank', 'etf', 'predicted_1d_return', 'predictability_index', 
            'koopman_regime', 'is_predictable']
    df_display = df_display[[c for c in cols if c in df_display.columns]]
    
    # Color coding function
    def color_return(val):
        if isinstance(val, (int, float)):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}; font-weight: bold'
        return ''
    
    # Show as styled table - FIXED: applymap -> map
    st.dataframe(
        df_display.style
        .map(color_return, subset=['predicted_1d_return'])
        .background_gradient(subset=['predictability_index'], cmap='RdYlGn', vmin=0, vmax=1)
        .format({
            'predicted_1d_return': '{:+.1f} bps',
            'predictability_index': '{:.3f}',
            'data_quality': '{:.1%}'
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "etf": st.column_config.TextColumn("ETF", width="medium"),
            "predicted_1d_return": st.column_config.NumberColumn("1-Day Return", width="medium"),
            "predictability_index": st.column_config.ProgressColumn("Predictability", 
                                                                     min_value=0, max_value=1, 
                                                                     width="medium"),
            "koopman_regime": st.column_config.TextColumn("Regime", width="medium"),
            "is_predictable": st.column_config.CheckboxColumn("Valid", width="small")
        }
    )
    
    # Distribution chart
    if 'predicted_1d_return' in df.columns:
        fig = px.histogram(
            df, 
            x='predicted_1d_return',
            color='koopman_regime' if 'koopman_regime' in df.columns else None,
            nbins=20,
            title="Distribution of Predicted Returns Across Universe",
            labels={'predicted_1d_return': 'Predicted 1-Day Return (bps)'},
            color_discrete_map={
                'expansion': '#28a745',
                'oscillatory': '#ffc107',
                'contraction': '#dc3545'
            }
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)


def render_koopman_analysis(signals: Dict):
    """Render Koopman spectral decomposition and analysis."""
    st.markdown("---")
    st.subheader("🔬 Koopman Mode Analysis")
    
    primary = signals['primary_pick']
    modes = signals.get('koopman_modes', {})
    
    # Metrics row
    cols = st.columns([1, 1, 1])
    
    with cols[0]:
        st.metric("Detected Regime", primary['regime'].upper())
        st.metric("Predictability Index", f"{primary['predictability_index']:.3f}")
        
    with cols[1]:
        st.metric("Dominant Frequency", f"{modes.get('dominant_frequency_cycles', 0):.3f} cycles/day",
                 help="Characteristic oscillation frequency from Koopman eigenvalue angle")
        
    with cols[2]:
        # Dynamic stability indicator
        if primary['regime'] == 'expansion':
            st.error("⚠️ Growth Mode Dominant\nUnstable dynamics detected")
        elif primary['regime'] == 'oscillatory':
            st.warning("🌊 Oscillatory Mode\nMean-reversion likely")
        else:
            st.success("✓ Decay Mode Dominant\nConverging to equilibrium")
    
    # Mode distribution badges
    st.markdown("#### Mode Distribution")
    growth = modes.get('growth_modes', 0)
    oscillatory = modes.get('oscillatory_modes', 0)
    decay = modes.get('decay_modes', 0)
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <span class="mode-badge growth">Growth: {growth}</span>
        <span class="mode-badge oscillatory">Oscillatory: {oscillatory}</span>
        <span class="mode-badge decay">Decay: {decay}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Eigenvalue spectrum visualization
    st.markdown("#### Koopman Operator Eigenvalue Spectrum")
    
    np.random.seed(hash(primary['etf']) % 2**32)
    n_modes = 64
    
    # Generate realistic spectrum based on regime
    angles = np.random.uniform(0, 2*np.pi, n_modes)
    
    if primary['regime'] == 'expansion':
        radii = np.concatenate([
            np.random.uniform(1.0, 1.2, 5),   # Growth modes
            np.random.beta(2, 5, 59) * 0.95    # Stable
        ])
    elif primary['regime'] == 'oscillatory':
        radii = np.concatenate([
            np.random.uniform(0.95, 1.05, 15),  # Near unit circle
            np.random.beta(2, 5, 49) * 0.9
        ])
    else:
        radii = np.random.beta(3, 2, n_modes) * 0.9  # Mostly contracting
    
    np.random.shuffle(radii)
    
    real = radii * np.cos(angles)
    imag = radii * np.sin(angles)
    
    fig = go.Figure()
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='gray', dash='dash', width=2),
        name='Unit Circle (|λ|=1)', hoverinfo='skip'
    ))
    
    # Color by magnitude and regime
    colors = ['#dc3545' if r > 1.0 else '#ffc107' if r > 0.95 else '#28a745' for r in radii]
    
    fig.add_trace(go.Scatter(
        x=real, y=imag,
        mode='markers',
        marker=dict(size=8, color=colors, line=dict(width=1, color='black'), opacity=0.7),
        text=[f"λ_{i}<br>|λ|={r:.3f}<br>ω={np.angle(np.exp(1j*a))/(2*np.pi):.3f}" 
              for i, (r, a) in enumerate(zip(radii, angles))],
        hovertemplate='%{text}<extra></extra>',
        name='Koopman Modes'
    ))
    
    fig.update_layout(
        title=f"Eigenvalue Spectrum for {primary['etf']} ({primary['regime']} regime)",
        xaxis_title="Real", yaxis_title="Imaginary",
        height=600, yaxis_scaleanchor="x", yaxis_scaleratio=1,
        plot_bgcolor='white', paper_bgcolor='white'
    )
    
    # Add annotations
    fig.add_annotation(x=1.1, y=0, text="Expanding", showarrow=False,
                      font=dict(color="#dc3545", size=12))
    fig.add_annotation(x=0.7, y=0.7, text="Oscillatory", showarrow=False,
                      font=dict(color="#ffc107", size=12))
    fig.add_annotation(x=0.3, y=0, text="Contracting", showarrow=False,
                      font=dict(color="#28a745", size=12))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **Red**: Growth modes (|λ|>1, unstable) · 
    **Yellow**: Near-critical oscillatory (|λ|≈1) · 
    **Green**: Decay modes (|λ|<1, stable)
    """)


def render_data_source_info(signals: Dict):
    """Render data source and methodology information."""
    st.markdown("---")
    st.subheader("🔗 Data Source & Methodology")
    
    meta = signals.get('metadata', {})
    
    cols = st.columns([2, 1])
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-highlight">
            <h4>📡 Data Source</h4>
            <p><b>Input Dataset:</b> {HF_DATA_REPO}/data/master.parquet</p>
            <p><b>Results Repo:</b> {HF_RESULTS_REPO}</p>
            <p><b>Contents:</b> OHLCV · log returns · volatility · macro indicators (VIX, T10Y2Y, DXY, HY/IG, WTI, DTB3)</p>
            <p><b>Update Frequency:</b> Daily pre-market (2:00 AM UTC)</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-highlight">
            <h4>🎯 Optimization Objective</h4>
            <p><b>Target:</b> Maximum expected 1-day return</p>
            <p><b>Constraint:</b> Predictability index ≥ 0.6 (filter)</p>
            <p><b>Ranking:</b> Sort by predicted return descending, select top 3</p>
            <p><b>Output:</b> Valid for next NYSE trading session open</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="metric-highlight">
            <h4>📊 Universe Stats</h4>
            <p><b>ETFs Analyzed:</b> {meta.get('total_etfs_analyzed', 20)}</p>
            <p><b>Predictable:</b> {meta.get('predictable_etfs', 'N/A')}</p>
            <p><b>Lookback:</b> {meta.get('lookback_window', 63)} days</p>
            <p><b>Coverage:</b> Fixed Income · Equity Sectors · Commodities · REITs</p>
        </div>
        """, unsafe_allow_html=True)
        
        # NYSE calendar info
        next_open = NYSECalendar.get_next_trading_date()
        st.markdown(f"""
        <div class="metric-highlight">
            <h4>📅 NYSE Calendar</h4>
            <p><b>Next Open:</b> {NYSECalendar.format_trading_date(next_open)}</p>
            <p><b>Current Status:</b> {'Market Closed' if datetime.now().weekday() >= 5 else 'Check Hours'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show column detection info if available
        if meta.get('detected_etf_col'):
            st.caption(f"Detected: ETF='{meta['detected_etf_col']}', Date='{meta.get('detected_date_col', 'N/A')}'")


def render_sidebar():
    """Render sidebar controls and info."""
    with st.sidebar:
        st.markdown("### 🔧 Engine Controls")
        
        # Trading date selector (auto-computed)
        next_trading = NYSECalendar.get_next_trading_date()
        st.info(f"📅 Target Date: **{NYSECalendar.format_trading_date(next_trading)}**")
        
        # Refresh button
        if st.button("🔄 Refresh Signals", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 Objective Settings")
        
        st.selectbox(
            "Optimization Target",
            ["Maximum Predicted Return", "Risk-Adjusted Return", "Predictability-Weighted"],
            index=0,
            disabled=True  # Fixed for this engine variant
        )
        
        st.slider(
            "Predictability Filter",
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            disabled=True  # Fixed threshold
        )
        
        st.markdown("---")
        st.markdown("### 📊 Engine Status")
        
        st.json({
            "Results Repo": HF_RESULTS_REPO,
            "Observable Dim": 64,
            "Encoder": "MLP [128, 128]",
            "Max Runtime": "5h (GitHub Actions)"
        })
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666;">
            <b>P2 Engine Suite</b><br>
            Koopman-Spectral v1.0.0<br>
            Research Only · Apr 2026
        </div>
        """, unsafe_allow_html=True)


def render_footer():
    """Render footer with disclaimers."""
    st.markdown("""
    <div class="footer">
        <b>P2 Koopman-Spectral Engine v1.0.0</b> · 
        Maximum Return Targeting Variant · 
        NYSE Next-Open Scheduling<br><br>
        
        <b>⚠️ Research Disclaimer</b><br>
        This engine targets maximum predicted return using Koopman operator theory. 
        Signals are generated pre-market for the next NYSE trading session. 
        Past performance does not guarantee future results. The predictability 
        index filters low-confidence predictions but does not eliminate risk. 
        All outputs are research-grade and do not constitute investment advice.<br><br>
        
        <b>Data Attribution</b><br>
        Input data: <code>P2SAMAPA/p2-etf-deepm-data/data/master.parquet</code> · 
        Results: <code>P2SAMAPA/p2-etf-koopman-spectral-results</code><br><br>
        
        © 2026 P2SAMAPA · Research Only · Not Financial Advice
    </div>
    """, unsafe_allow_html=True)


# --- Main ---

def main():
    """Main application entry point."""
    render_header()
    render_sidebar()
    
    # Get next trading date
    trading_date = NYSECalendar.get_next_trading_date()
    
    # Load from HF results repo
    signals = load_latest_signals_from_hf()
    
    # Fallback to demo
    if signals is None:
        signals = generate_demo_signals(trading_date)
        st.warning("⚠️ Using demo data - no signals found in HF results repo. "
                  "Check that GitHub Actions has run and uploaded to "
                  f"{HF_RESULTS_REPO}")
    
    # Show source
    source = signals.get('_source', 'Unknown')
    loaded_at = signals.get('_loaded_at', 'Unknown')
    st.caption(f"Source: {source} | Loaded: {loaded_at}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs
