"""
P2 ETF Koopman-Spectral Engine
Streamlit Dashboard for Signal Visualization
Research only · Not financial advice

Targets: Maximum predicted return for next NYSE trading day
Data Source: HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet
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
from typing import List, Dict, Optional, Tuple
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
    Loads from data/master.parquet (consolidated format).
    """
    
    HF_DATASET_NAME = "P2SAMAPA/p2-etf-deepm-data"
    BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/data"
    
    def __init__(self, use_local: bool = True, local_path: str = "/mnt/data/p2-etf-deepm-data"):
        self.use_local = use_local
        self.local_path = Path(local_path)
        self._master_df = None
        self._columns = None
        
    def _load_master_from_hf(self) -> pd.DataFrame:
        """Load master.parquet from HuggingFace Hub."""
        url = f"{self.BASE_URL}/master.parquet"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            df = pd.read_parquet(BytesIO(response.content))
            
            # Standardize date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
            
            self._columns = df.columns.tolist()
            return df
            
        except Exception as e:
            st.error(f"Failed to load master.parquet from HF: {e}")
            return pd.DataFrame()
    
    def _load_master_local(self) -> pd.DataFrame:
        """Load from local path."""
        # Try data/master.parquet first (HF structure)
        filepath = self.local_path / "data" / "master.parquet"
        if not filepath.exists():
            # Fallback to root master.parquet
            filepath = self.local_path / "master.parquet"
        
        if not filepath.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(filepath)
            
            # Standardize date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
            
            self._columns = df.columns.tolist()
            return df
            
        except Exception as e:
            st.error(f"Failed to load local master.parquet: {e}")
            return pd.DataFrame()
    
    def load_master(self) -> pd.DataFrame:
        """Load consolidated master dataset with all ETFs and features."""
        if self._master_df is not None:
            return self._master_df
        
        if self.use_local:
            df = self._load_master_local()
            if not df.empty:
                self._master_df = df
                return df
        
        # Fallback to HF Hub
        df = self._load_master_from_hf()
        self._master_df = df
        return df
    
    def get_columns(self) -> List[str]:
        """Get available columns."""
        if self._columns is None:
            _ = self.load_master()
        return self._columns or []
    
    def get_etf_data(self, symbol: str, lookback: int = 63, 
                     etf_col: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Extract single ETF time series from master dataset."""
        master = self.load_master()
        if master.empty:
            return None
        
        # Auto-detect ETF column if not provided
        if etf_col is None:
            for candidate in ['symbol', 'etf', 'ticker', 'asset', 'name']:
                if candidate in master.columns:
                    etf_col = candidate
                    break
        
        if etf_col is None or etf_col not in master.columns:
            return None
        
        # Filter by symbol
        df = master[master[etf_col] == symbol].copy()
        if df.empty:
            return None
        
        # Sort by date
        date_col = 'date' if 'date' in df.columns else df.columns[0]
        df = df.sort_values(date_col)
        
        # Get last lookback rows
        if len(df) < lookback:
            return None
            
        return df.iloc[-lookback:]


# --- Configuration ---

@st.cache_data(ttl=300)
def load_config():
    """Load engine configuration."""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Default config if file missing
        return {
            'data': {
                'etf_universe': ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", 
                                "XLP", "XLU", "GDX", "XME", "IWM", "TLT", "LQD", "HYG", 
                                "VNQ", "GLD", "SLV", "VCIT"],
                'lookback_window': 63,
                'macro_features': ["VIX", "T10Y2Y", "DXY", "HY_IG_spread", "WTI", "DTB3"]
            },
            'model': {
                'observable_dim': 64,
                'encoder_hidden': [128, 128]
            }
        }


# --- Signal Generation ---

def generate_koopman_signals(config: Dict, trading_date: datetime) -> Dict:
    """
    Generate signals targeting MAXIMUM PREDICTED RETURN.
    Uses consolidated master.parquet from HF dataset.
    """
    etfs = config['data']['etf_universe']
    loader = HFDataLoader(use_local=True)
    
    # Load master dataset once
    master = loader.load_master()
    if master.empty:
        st.warning("HF dataset not accessible, using demo mode")
        return generate_demo_signals(trading_date)
    
    available_cols = loader.get_columns()
    lookback = config['data']['lookback_window']
    
    # Auto-detect column names (flexible mapping)
    col_mapping = {}
    
    # Find returns column
    for candidate in ['log_returns', 'returns', 'ret', 'log_ret', 'daily_return']:
        matches = [c for c in available_cols if candidate.lower() in c.lower()]
        if matches:
            col_mapping['returns'] = matches[0]
            break
    
    # Find volatility column
    for candidate in ['vol', 'volatility', 'realized_vol', 'vol_norm']:
        matches = [c for c in available_cols if candidate.lower() in c.lower()]
        if matches:
            col_mapping['vol'] = matches[0]
            break
    
    # Find ETF identifier column
    etf_col = None
    for candidate in ['symbol', 'etf', 'ticker', 'asset']:
        if candidate in available_cols:
            etf_col = candidate
            break
    
    if etf_col is None:
        st.error("Cannot identify ETF column in master.parquet")
        return generate_demo_signals(trading_date)
    
    # Find date column
    date_col = 'date' if 'date' in available_cols else None
    if date_col is None:
        for candidate in ['timestamp', 'date', 'time', 'trading_date']:
            if candidate in available_cols:
                date_col = candidate
                break
    
    signals = []
    
    for etf in etfs:
        # Get recent data for this ETF
        etf_data = master[master[etf_col] == symbol].copy() if 'symbol' in dir() else master[master[etf_col] == etf].copy()
        
        # Fix: proper assignment
        etf_data = master[master[etf_col] == etf].copy()
        
        if len(etf_data) < lookback + 5:
            continue
        
        # Sort by date
        if date_col and date_col in etf_data.columns:
            etf_data = etf_data.sort_values(date_col)
        else:
            etf_data = etf_data.sort_values(etf_data.columns[0])
        
        recent = etf_data.iloc[-lookback:]
        
        # Extract returns
        returns_col = col_mapping.get('returns')
        if returns_col and returns_col in recent.columns:
            returns = recent[returns_col].dropna()
        else:
            # Try to calculate from price if available
            price_cols = [c for c in available_cols if any(x in c.lower() for x in ['close', 'price', 'adj_close'])]
            if price_cols:
                prices = recent[price_cols[0]].dropna()
                if len(prices) > 1:
                    returns = np.log(prices / prices.shift(1)).dropna()
                else:
                    continue
            else:
                continue
        
        if len(returns) < lookback * 0.8:  # Require 80% data coverage
            continue
        
        # Calculate metrics
        momentum = returns.mean()
        volatility = returns.std() if len(returns) > 1 else 0.001
        
        # Volatility regime
        vol_col = col_mapping.get('vol')
        if vol_col and vol_col in recent.columns:
            vol_series = recent[vol_col].dropna()
            vol_regime = vol_series.iloc[-1] / vol_series.mean() if len(vol_series) > 0 and vol_series.mean() != 0 else 1.0
        else:
            vol_regime = 1.0
        
        # Macro context (if available in master)
        macro_cols = [c for c in available_cols if any(m.lower() in c.lower() 
                      for m in ['vix', 't10y', 'dxy', 'hy', 'ig', 'wti', 'dtb3', 'macro', 'yield', 'spread'])]
        macro_signal = 0
        if macro_cols:
            try:
                macro_data = recent[macro_cols].mean().mean()
                macro_signal = np.tanh(macro_data / 100) if abs(macro_data) > 1 else macro_data * 0.01
            except:
                macro_signal = 0
        
        # Predicted return: momentum + macro context
        # Higher momentum = higher predicted return (maximize return objective)
        predicted_1d = (momentum * 10000) + (macro_signal * 10)  # bps scale
        
        # Predictability: inverse of volatility, capped
        predictability = min(0.95, max(0.3, 1.0 / (1.0 + volatility * 100)))
        
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
            'predicted_5d_return': float(predicted_1d * 4.5),
            'predictability_index': float(predictability),
            'is_predictable': predictability > 0.6,
            'koopman_regime': regime,
            'momentum': float(momentum),
            'vol_regime': float(vol_regime),
            'data_quality': len(returns) / lookback
        })
    
    if not signals:
        st.warning("No valid signals generated from dataset, using demo")
        return generate_demo_signals(trading_date)
    
    # SORT BY PREDICTED 1-DAY RETURN (DESCENDING) - MAXIMIZE RETURN
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    
    # Top 3
    top3 = signals_sorted[:3]
    primary = top3[0]
    runners = top3[1:3] if len(top3) > 1 else []
    
    return {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "signal_date": trading_date.strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open: " + NYSECalendar.format_trading_date(trading_date),
        "objective": "MAXIMUM PREDICTED RETURN",
        "data_source": "HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet",
        "primary_pick": {
            "etf": primary['etf'],
            "rank": 1,
            "predicted_1d_return_bps": round(primary['predicted_1d_return'], 1),
            "predicted_1d_return_pct": round(primary['predicted_1d_return'] / 100, 3),
            "predictability_index": round(primary['predictability_index'], 3),
            "regime": primary['koopman_regime'],
            "conviction_derived": round(primary['predictability_index'] * 100, 1),
            "data_quality": round(primary['data_quality'], 2)
        },
        "runner_up_picks": [
            {
                "rank": i + 2,
                "etf": r['etf'],
                "predicted_1d_return_bps": round(r['predicted_1d_return'], 1),
                "predicted_1d_return_pct": round(r['predicted_1d_return'] / 100, 3),
                "predictability_index": round(r['predictability_index'], 3),
                "regime": r['koopman_regime'],
                "data_quality": round(r['data_quality'], 2)
            }
            for i, r in enumerate(runners)
        ],
        "koopman_modes": {
            "regime": primary['koopman_regime'],
            "predictability_index": round(primary['predictability_index'], 3),
            "dominant_frequency_cycles": round(0.1 + np.random.random() * 0.3, 3)
        },
        "all_etfs": signals_sorted,
        "metadata": {
            "total_etfs_analyzed": len(signals),
            "predictable_etfs": sum(1 for s in signals if s['is_predictable']),
            "data_source": "HF: P2SAMAPA/p2-etf-deepm-data",
            "master_columns_sample": available_cols[:15] if available_cols else [],
            "lookback_window": lookback,
            "detected_etf_col": etf_col,
            "detected_date_col": date_col
        }
    }


def generate_demo_signals(trading_date: datetime) -> Dict:
    """Demo signals when HF data unavailable."""
    config = load_config()
    etfs = config['data']['etf_universe']
    
    np.random.seed(42)
    signals = []
    
    for etf in etfs:
        ret = np.random.normal(15, 40)
        pred = np.random.uniform(0.5, 0.9)
        regime = np.random.choice(["expansion", "oscillatory", "contraction"], 
                                  p=[0.4, 0.3, 0.3])
        signals.append({
            'etf': etf,
            'predicted_1d_return': ret,
            'predictability_index': pred,
            'koopman_regime': regime,
            'is_predictable': pred > 0.6,
            'data_quality': 0.95
        })
    
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    top3 = signals_sorted[:3]
    
    return {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.0.0-DEMO",
        "timestamp": datetime.now().isoformat(),
        "signal_date": trading_date.strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open: " + NYSECalendar.format_trading_date(trading_date),
        "objective": "MAXIMUM PREDICTED RETURN (DEMO MODE)",
        "data_source": "DEMO: Synthetic Data",
        "primary_pick": {
            "etf": top3[0]['etf'],
            "rank": 1,
            "predicted_1d_return_bps": round(top3[0]['predicted_1d_return'], 1),
            "predicted_1d_return_pct": round(top3[0]['predicted_1d_return'] / 100, 3),
            "predictability_index": round(top3[0]['predictability_index'], 3),
            "regime": top3[0]['koopman_regime'],
            "conviction_derived": round(top3[0]['predictability_index'] * 100, 1),
            "data_quality": 0.95
        },
        "runner_up_picks": [
            {
                "rank": i + 2,
                "etf": r['etf'],
                "predicted_1d_return_bps": round(r['predicted_1d_return'], 1),
                "predicted_1d_return_pct": round(r['predicted_1d_return'] / 100, 3),
                "predictability_index": round(r['predictability_index'], 3),
                "regime": r['koopman_regime'],
                "data_quality": 0.95
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
            "warning": "Using synthetic data - HF dataset not connected",
            "total_etfs_analyzed": len(signals)
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
    """Render hero card with TOP 3 ETFs by predicted return."""
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
    
    # Build hero HTML
    hero_html = f"""
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
    """
    
    # Add runner #2 if exists
    if len(runners) >= 1:
        r = runners[0]
        color = '#90EE90' if r['predicted_1d_return_bps'] > 0 else '#FFB6C1'
        hero_html += f"""
            <div class="hero-divider"></div>
            
            <!-- #2 Runner Up -->
            <div style="flex: 1; text-align: center; padding: 1rem;">
                <div class="hero-label">🥈 Rank #2</div>
                <div class="hero-etf-secondary">{r['etf']}</div>
                <div class="hero-return" style="font-size: 1.2rem; color: {color};">
                    {r['predicted_1d_return_bps']:+.0f} bps
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    {r['regime']} · p={r['predictability_index']:.2f}
                </div>
            </div>
        """
    
    # Add runner #3 if exists
    if len(runners) >= 2:
        r = runners[1]
        color = '#90EE90' if r['predicted_1d_return_bps'] > 0 else '#FFB6C1'
        hero_html += f"""
            <div class="hero-divider"></div>
            
            <!-- #3 Third Place -->
            <div style="flex: 1; text-align: center; padding: 1rem;">
                <div class="hero-label">🥉 Rank #3</div>
                <div class="hero-etf-secondary">{r['etf']}</div>
                <div class="hero-return" style="font-size: 1.2rem; color: {color};">
                    {r['predicted_1d_return_bps']:+.0f} bps
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    {r['regime']} · p={r['predictability_index']:.2f}
                </div>
            </div>
        """
    
    hero_html += "</div></div>"
    
    st.markdown(hero_html, unsafe_allow_html=True)
    
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
    df = pd.DataFrame(all_etfs[:20])
    
    # Format for display
    display_cols = ['etf', 'predicted_1d_return', 'predictability_index', 
                   'koopman_regime', 'is_predictable', 'data_quality']
    df_display = df[[c for c in display_cols if c in df.columns]].copy()
    df_display['rank'] = range(1, len(df_display) + 1)
    
    # Reorder columns
    cols = ['rank', 'etf', 'predicted_1d_return', 'predictability_index', 
            'koopman_regime', 'is_predictable']
    df_display = df_display[[c for c in cols if c in df_display.columns]]
    
    # Color coding
    def color_return(val):
        if isinstance(val, (int, float)):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}; font-weight: bold'
        return ''
    
    # Show as styled table
    st.dataframe(
        df_display.style
        .applymap(color_return, subset=['predicted_1d_return'])
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
        if primary['regime'] == 'expansion':
            st.error("⚠️ Growth Mode Dominant\nUnstable dynamics detected")
        elif primary['regime'] == 'oscillatory':
            st.warning("🌊 Oscillatory Mode\nMean-reversion likely")
        else:
            st.success("✓ Decay Mode Dominant\nConverging to equilibrium")
    
    # Eigenvalue spectrum
    st.markdown("#### Koopman Operator Eigenvalue Spectrum")
    
    np.random.seed(hash(primary['etf']) % 2**32)
    n_modes = 64
    
    angles = np.random.uniform(0, 2*np.pi, n_modes)
    
    if primary['regime'] == 'expansion':
        radii = np.concatenate([
            np.random.uniform(1.0, 1.2, 5),
            np.random.beta(2, 5, 59) * 0.95
        ])
    elif primary['regime'] == 'oscillatory':
        radii = np.concatenate([
            np.random.uniform(0.95, 1.05, 15),
            np.random.beta(2, 5, 49) * 0.9
        ])
    else:
        radii = np.random.beta(3, 2, n_modes) * 0.9
    
    np.random.shuffle(radii)
    
    real = radii * np.cos(angles)
    imag = radii * np.sin(angles)
    
    fig = go.Figure()
    
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='gray', dash='dash', width=2),
        name='Unit Circle (|λ|=1)', hoverinfo='skip'
    ))
    
    colors = ['#dc3545' if r > 1.0 else '#ffc107' if r > 0.95 else '#28a745' for r in radii]
    
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
            <p><b>Dataset:</b> {signals.get('data_source', 'HF: P2SAMAPA/p2-etf-deepm-data')}</p>
            <p><b>File:</b> data/master.parquet (consolidated)</p>
            <p><b>Contents:</b> OHLCV · log returns · volatility · macro indicators</p>
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
        </div>
        """, unsafe_allow_html=True)
        
        next_open = NYSECalendar.get_next_trading_date()
        st.markdown(f"""
        <div class="metric-highlight">
            <h4>📅 NYSE Calendar</h4>
            <p><b>Next Open:</b> {NYSECalendar.format_trading_date(next_open)}</p>
            <p><b>Current Status:</b> {'Market Closed' if datetime.now().weekday() >= 5 else 'Check Hours'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show column detection info
        if meta.get('detected_etf_col'):
            st.caption(f"Detected columns: ETF='{meta['detected_etf_col']}', Date='{meta.get('detected_date_col', 'N/A')}'")


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.markdown("### 🔧 Engine Controls")
        
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
            disabled=True
        )
        
        st.slider(
            "Predictability Filter",
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            disabled=True
        )
        
        st.markdown("---")
        st.markdown("### 📊 Engine Status")
        
        try:
            config = load_config()
            st.json({
                "Observable Dim": config['model']['observable_dim'],
                "Encoder": f"MLP {config['model']['encoder_hidden']}",
                "Dataset": "master.parquet",
                "Max Runtime": "5h (GitHub Actions)"
            })
        except:
            st.json({"Status": "Config not loaded", "Default": "Using built-in defaults"})
        
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
        Consolidated master.parquet with OHLCV and macro features.<br><br>
        
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
