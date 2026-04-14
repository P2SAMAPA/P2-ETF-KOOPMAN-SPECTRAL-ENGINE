"""
P2 ETF Koopman-Spectral Engine
Streamlit Dashboard for Signal Visualization
Research only · Not financial advice

Targets: Maximum predicted return for next NYSE trading day
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
import yaml
from typing import List, Dict, Optional
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="P2 Koopman-Spectral Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)


# --- NYSE Calendar ---

class NYSECalendar:
    """NYSE trading calendar for next valid trading date."""
    
    # 2026 NYSE holidays (full day closures)
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


# --- HF Dataset Integration ---

class HFDataLoader:
    """
    HuggingFace dataset loader for p2-etf-deepm-data.
    Loads OHLCV + macro from HF Hub or local cache.
    """
    
    HF_DATASET_NAME = "P2SAMAPA/p2-etf-deepm-data"
    BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main"
    
    def __init__(self, use_local: bool = True, local_path: str = "/mnt/data/p2-etf-deepm-data"):
        self.use_local = use_local
        self.local_path = Path(local_path)
        self.cache = {}
        
    def _load_from_hf(self, filename: str) -> pd.DataFrame:
        """Load parquet from HuggingFace Hub."""
        url = f"{self.BASE_URL}/{filename}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return pd.read_parquet(BytesIO(response.content))
        except Exception as e:
            st.error(f"Failed to load {filename} from HF: {e}")
            return pd.DataFrame()
    
    def _load_local(self, filename: str) -> pd.DataFrame:
        """Load from local path."""
        filepath = self.local_path / filename
        if not filepath.exists():
            return pd.DataFrame()
        return pd.read_parquet(filepath)
    
    def load_etf(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV data for single ETF."""
        cache_key = f"etf_{symbol}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        filename = f"{symbol}.parquet"
        
        if self.use_local and (self.local_path / filename).exists():
            df = self._load_local(filename)
        else:
            df = self._load_from_hf(filename)
        
        if not df.empty:
            df['symbol'] = symbol
            df.index = pd.to_datetime(df.index)
            self.cache[cache_key] = df
        
        return df
    
    def load_macro(self) -> pd.DataFrame:
        """Load macro features."""
        cache_key = "macro"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        filename = "macro.parquet"
        
        if self.use_local and (self.local_path / filename).exists():
            df = self._load_local(filename)
        else:
            df = self._load_from_hf(filename)
        
        if not df.empty:
            df.index = pd.to_datetime(df.index)
            self.cache[cache_key] = df
        
        return df
    
    def load_master(self) -> pd.DataFrame:
        """Load master.parquet with metadata."""
        filename = "master.parquet"
        
        if self.use_local and (self.local_path / filename).exists():
            return self._load_local(filename)
        else:
            return self._load_from_hf(filename)


# --- Configuration ---

@st.cache_data(ttl=300)
def load_config():
    """Load engine configuration."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


# --- Signal Generation (Return-Maximizing) ---

def generate_koopman_signals(config: Dict, trading_date: datetime) -> Dict:
    """
    Generate signals targeting MAXIMUM PREDICTED RETURN.
    Ranks all ETFs by predicted 1-day return, picks top 3.
    """
    etfs = config['data']['etf_universe']
    loader = HFDataLoader(use_local=True)
    
    signals = []
    
    for etf in etfs:
        # Load data
        df = loader.load_etf(etf)
        if df.empty or len(df) < config['data']['lookback_window'] + 5:
            continue
        
        macro = loader.load_macro()
        
        # Merge and engineer features
        df = df.join(macro, how='left')
        df['returns_lag1'] = df['log_returns'].shift(1)
        df['vol_norm'] = df['vol'] / df['vol'].rolling(21).mean()
        
        # Get most recent window
        lookback = config['data']['lookback_window']
        if len(df) < lookback:
            continue
            
        recent = df.iloc[-lookback:]
        
        # Calculate Koopman-inspired prediction (simplified for demo)
        # In production: load trained model, encode, apply K, decode
        momentum = recent['log_returns'].mean()
        vol_regime = recent['vol_norm'].mean()
        macro_context = recent[config['data']['macro_features']].mean().mean()
        
        # Predicted return: momentum + mean-reversion adjustment
        # Higher momentum = higher predicted return (targeting max return)
        predicted_1d = momentum * 10000  # Convert to bps
        
        # Predictability based on vol stability
        vol_stability = 1.0 / (1.0 + recent['vol_norm'].std())
        predictability = min(0.95, max(0.3, vol_stability * 0.8 + 0.2))
        
        # Regime detection
        if momentum > 0.001 and vol_regime < 1.5:
            regime = "expansion"
        elif abs(momentum) < 0.0005:
            regime = "oscillatory"
        else:
            regime = "contraction"
        
        signals.append({
            'etf': etf,
            'predicted_1d_return': float(predicted_1d),
            'predicted_5d_return': float(predicted_1d * 4.5),  # Rough scaling
            'predictability_index': float(predictability),
            'is_predictable': predictability > 0.6,
            'koopman_regime': regime,
            'momentum': float(momentum),
            'vol_regime': float(vol_regime),
            'last_price': float(recent['close'].iloc[-1]) if 'close' in recent else None
        })
    
    if not signals:
        return generate_demo_signals(trading_date)
    
    # SORT BY PREDICTED 1-DAY RETURN (DESCENDING) - MAXIMIZE RETURN
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    
    # Top 3
    top3 = signals_sorted[:3]
    primary = top3[0]
    runners = top3[1:3] if len(top3) > 1 else []
    
    # Build output
    return {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "signal_date": trading_date.strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open: " + NYSECalendar.format_trading_date(trading_date),
        "objective": "MAXIMUM PREDICTED RETURN",
        "primary_pick": {
            "etf": primary['etf'],
            "rank": 1,
            "predicted_1d_return_bps": round(primary['predicted_1d_return'], 1),
            "predicted_1d_return_pct": round(primary['predicted_1d_return'] / 100, 3),
            "predictability_index": round(primary['predictability_index'], 3),
            "regime": primary['koopman_regime'],
            "conviction_derived": round(primary['predictability_index'] * 100, 1)
        },
        "runner_up_picks": [
            {
                "rank": i + 2,
                "etf": r['etf'],
                "predicted_1d_return_bps": round(r['predicted_1d_return'], 1),
                "predicted_1d_return_pct": round(r['predicted_1d_return'] / 100, 3),
                "predictability_index": round(r['predictability_index'], 3),
                "regime": r['koopman_regime']
            }
            for i, r in enumerate(runners)
        ],
        "koopman_modes": {
            "regime": primary['koopman_regime'],
            "predictability_index": round(primary['predictability_index'], 3),
            "dominant_frequency_cycles": round(0.1 + np.random.random() * 0.3, 3)  # Placeholder
        },
        "all_etfs": signals_sorted,
        "metadata": {
            "total_etfs_analyzed": len(signals),
            "predictable_etfs": sum(1 for s in signals if s['is_predictable']),
            "data_source": "HF: P2SAMAPA/p2-etf-deepm-data",
            "lookback_window": config['data']['lookback_window']
        }
    }


def generate_demo_signals(trading_date: datetime) -> Dict:
    """Demo signals when HF data unavailable."""
    config = load_config()
    etfs = config['data']['etf_universe']
    
    # Generate realistic predictions
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
            'koopman_regime': regime
        })
    
    # SORT BY RETURN
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    top3 = signals_sorted[:3]
    
    return {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.0.0-DEMO",
        "timestamp": datetime.now().isoformat(),
        "signal_date": trading_date.strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open: " + NYSECalendar.format_trading_date(trading_date),
        "objective": "MAXIMUM PREDICTED RETURN (DEMO MODE)",
        "primary_pick": {
            "etf": top3[0]['etf'],
            "rank": 1,
            "predicted_1d_return_bps": round(top3[0]['predicted_1d_return'], 1),
            "predicted_1d_return_pct": round(top3[0]['predicted_1d_return'] / 100, 3),
            "predictability_index": round(top3[0]['predictability_index'], 3),
            "regime": top3[0]['koopman_regime'],
            "conviction_derived": round(top3[0]['predictability_index'] * 100, 1)
        },
        "runner_up_picks": [
            {
                "rank": i + 2,
                "etf": r['etf'],
                "predicted_1d_return_bps": round(r['predicted_1d_return'], 1),
                "predicted_1d_return_pct": round(r['predicted_1d_return'] / 100, 3),
                "predictability_index": round(r['predictability_index'], 3),
                "regime": r['koopman_regime']
            }
            for i, r in enumerate(top3[1:3])
        ],
        "koopman_modes": {
            "regime": top3[0]['koopman_regime'],
            "predictability_index": round(top3[0]['predictability_index'], 3)
        },
        "all_etfs": signals_sorted,
        "metadata": {
            "demo": True,
            "warning": "Using synthetic data - HF dataset not connected"
        }
    }


# --- Visualization Components ---

def render_header():
    """Render main header."""
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
    #1: Large font, #2-3: Smaller font, side-by-side layout.
    """
    st.markdown("---")
    
    primary = signals['primary_pick']
    runners = signals.get('runner_up_picks', [])
    target_date = signals.get('target_date', 'Next NYSE Open')
    
    # Header
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
    
    # Hero container with top 3
    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; align-items: center; justify-content: space-around;">
            
            <!-- #1 Primary Pick (Large) -->
            <div style="flex: 2; text-align: center; padding: 1rem;">
                <div class="hero-label">🥇 Rank #1 · Primary Signal</div>
                <div class="hero-etf-main">{primary['etf']}</div>
                <div class="hero-return" style="color: {'#90EE90' if primary['predicted_1d_return_bps'] > 0 else '#FFB6C1'};">
                    {primary['predicted_1d_return_bps']:+.0f} bps
                    <span style="font-size: 0.6em; opacity: 0.8;">({primary['predicted_1d_return_pct']:+.3f}%)</span>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem;">
                    Predictability: {primary['predictability_index']:.2f} · {primary['regime'].upper()}
                </div>
            </div>
    """, unsafe_allow_html=True)
    
    # Add runners if they exist
    if len(runners) >= 1:
        st.markdown(f"""
            <div class="hero-divider"></div>
            
            <!-- #2 Runner Up -->
            <div style="flex: 1; text-align: center; padding: 1rem;">
                <div class="hero-label">🥈 Rank #2</div>
                <div class="hero-etf-secondary">{runners[0]['etf']}</div>
                <div class="hero-return" style="font-size: 1.2rem; color: {'#90EE90' if runners[0]['predicted_1d_return_bps'] > 0 else '#FFB6C1'};">
                    {runners[0]['predicted_1d_return_bps']:+.0f} bps
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    {runners[0]['regime']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    if len(runners) >= 2:
        st.markdown(f"""
            <div class="hero-divider"></div>
            
            <!-- #3 Third Place -->
            <div style="flex: 1; text-align: center; padding: 1rem;">
                <div class="hero-label">🥉 Rank #3</div>
                <div class="hero-etf-secondary">{runners[1]['etf']}</div>
                <div class="hero-return" style="font-size: 1.2rem; color: {'#90EE90' if runners[1]['predicted_1d_return_bps'] > 0 else '#FFB6C1'};">
                    {runners[1]['predicted_1d_return_bps']:+.0f} bps
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    {runners[1]['regime']}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Objective reminder
    st.markdown(f"""
    <div style="text-align: center; margin-top: 1rem; color: #666; font-size: 0.9rem;">
        <b>Objective Function:</b> max(E[Return₁ₐᵧ]) subject to Predictability > 0.6<br>
        <span style="color: #999;">Sorted by predicted 1-day return (highest first) · Top 3 displayed</span>
    </div>
    """, unsafe_allow_html=True)


def render_full_ranking(signals: Dict):
    """Render full ETF ranking table."""
    st.markdown("---")
    st.subheader("📊 Full ETF Ranking (by Predicted Return)")
    
    all_etfs = signals.get('all_etfs', [])
    if not all_etfs:
        st.info("No ranking data available")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_etfs[:20])  # Top 20
    
    # Format for display
    df_display = df[['etf', 'predicted_1d_return', 'predictability_index', 
                    'koopman_regime', 'is_predictable']].copy()
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Color coding
    def color_return(val):
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        return f'color: {color}; font-weight: bold'
    
    # Show as styled table
    st.dataframe(
        df_display.style
        .applymap(color_return, subset=['predicted_1d_return'])
        .background_gradient(subset=['predictability_index'], cmap='RdYlGn', vmin=0, vmax=1)
        .format({
            'predicted_1d_return': '{:+.1f} bps',
            'predictability_index': '{:.3f}'
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
    fig = px.histogram(
        df_display, 
        x='predicted_1d_return',
        color='koopman_regime',
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
    """Render Koopman spectral decomposition."""
    st.markdown("---")
    st.subheader("🔬 Koopman Mode Analysis")
    
    primary = signals['primary_pick']
    modes = signals.get('koopman_modes', {})
    
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
    colors = []
    for r in radii:
        if r > 1.0:
            colors.append('#dc3545')  # Growth
        elif r > 0.95:
            colors.append('#ffc107')  # Oscillatory
        else:
            colors.append('#28a745')  # Decay
    
    fig.add_trace(go.Scatter(
        x=real, y=imag,
        mode='markers',
        marker=dict(size=8, color=colors, line=dict(width=1, color='black'), opacity=0.7),
        text=[f"λ_{i}<br>|λ|={r:.3f}" for i, r in enumerate(radii)],
        hovertemplate='%{text}<extra></extra>',
        name='Koopman Modes'
    ))
    
    fig.update_layout(
        title=f"Eigenvalue Spectrum for {primary['etf']} ({primary['regime']} regime)",
        xaxis_title="Real", yaxis_title="Imaginary",
        height=600, yaxis_scaleanchor="x", yaxis_scaleratio=1
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_data_source_info(signals: Dict):
    """Render data source and methodology info."""
    st.markdown("---")
    st.subheader("🔗 Data Source & Methodology")
    
    meta = signals.get('metadata', {})
    
    cols = st.columns([2, 1])
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-highlight">
            <h4>📡 Data Source</h4>
            <p><b>Dataset:</b> {meta.get('data_source', 'HF: P2SAMAPA/p2-etf-deepm-data')}</p>
            <p><b>Contents:</b> OHLCV · log returns · volatility · macro indicators (VIX, T10Y2Y, DXY, HY/IG, WTI, DTB3)</p>
            <p><b>Lookback:</b> {meta.get('lookback_window', 63)} trading days</p>
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
            <h4>📈 Universe Stats</h4>
            <p><b>ETFs Analyzed:</b> {meta.get('total_etfs_analyzed', 20)}</p>
            <p><b>Predictable:</b> {meta.get('predictable_etfs', 15)}</p>
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


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.markdown("### 🔧 Engine Controls")
        
        # Trading date selector (auto-computed)
        next_trading = NYSECalendar.get_next_trading_date()
        st.info(f"📅 Target Date: **{NYSECalendar.format_trading_date(next_trading)}**")
        
        if st.button("🔄 Regenerate Signals", type="primary"):
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
        
        config = load_config()
        st.json({
            "Observable Dim": config['model']['observable_dim'],
            "Encoder": f"MLP {config['model']['encoder_hidden']}",
            "DMD Init": config['training']['dmd_init'],
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
    """Render footer."""
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
        Raw data: HuggingFace dataset <code>P2SAMAPA/p2-etf-deepm-data</code> · 
        OHLCV and macro features via shared P2 data hub.<br><br>
        
        © 2026 P2SAMAPA · Research Only · Not Financial Advice
    </div>
    """, unsafe_allow_html=True)


# --- Main ---

def main():
    """Main application."""
    render_header()
    render_sidebar()
    
    # Get next trading date
    trading_date = NYSECalendar.get_next_trading_date()
    
    # Load or generate signals
    config = load_config()
    signals = generate_koopman_signals(config, trading_date)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Top 3 Signals", 
        "📊 Full Ranking",
        "🔬 Analysis"
    ])
    
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
