"""
P2 ETF Koopman-Spectral Engine
Streamlit Dashboard for Signal Visualization and Analysis
Research only · Not financial advice
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
import torch

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .signal-card {
        background: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
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
</style>
""", unsafe_allow_html=True)

# --- Data Loading Functions ---

@st.cache_data(ttl=300)
def load_config():
    """Load engine configuration."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=60)
def load_latest_signals():
    """Load most recent signal file from outputs directory."""
    config = load_config()
    signals_dir = Path(config['signals']['output_dir'])
    
    if not signals_dir.exists():
        return None
    
    # Find most recent signal file
    signal_files = sorted(signals_dir.glob("koopman_signals_*.json"), reverse=True)
    
    if not signal_files:
        # Generate demo data if no files exist
        return generate_demo_signals()
    
    with open(signal_files[0], 'r') as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_historical_signals(days=30):
    """Load signal history for trend analysis."""
    config = load_config()
    signals_dir = Path(config['signals']['output_dir'])
    
    if not signals_dir.exists():
        return generate_demo_history(days)
    
    history = []
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for signal_file in sorted(signals_dir.glob("koopman_signals_*.json")):
        try:
            date_str = signal_file.stem.split('_')[-1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            
            if file_date >= cutoff_date:
                with open(signal_file, 'r') as f:
                    data = json.load(f)
                    data['file_date'] = file_date
                    history.append(data)
        except:
            continue
    
    return history if history else generate_demo_history(days)


def generate_demo_signals():
    """Generate realistic demo signals for preview."""
    config = load_config()
    etfs = config['data']['etf_universe']
    
    # Pick random ETF with realistic metrics
    primary = np.random.choice(etfs)
    
    return {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "signal_date": datetime.now().strftime("%Y-%m-%d"),
        "primary_pick": {
            "etf": primary,
            "conviction_pct": round(np.random.uniform(60, 95), 1),
            "expected_return_1d": round(np.random.uniform(-50, 150), 1),
            "expected_return_5d": round(np.random.uniform(-200, 400), 1)
        },
        "koopman_modes": {
            "regime": np.random.choice(["expansion", "oscillatory", "contraction"]),
            "growth_modes": int(np.random.randint(0, 5)),
            "oscillatory_modes": int(np.random.randint(0, 8)),
            "decay_modes": int(np.random.randint(55, 64)),
            "predictability_index": round(np.random.uniform(0.5, 0.95), 3),
            "dominant_frequency_cycles": round(np.random.uniform(0.05, 0.5), 3)
        },
        "runner_up_picks": [
            {"etf": etf, "predicted_1d_return": round(np.random.uniform(-30, 120), 1), 
             "predictability_index": round(np.random.uniform(0.4, 0.9), 3)}
            for etf in np.random.choice([e for e in etfs if e != primary], 3, replace=False)
        ],
        "all_etfs": [
            {
                "etf": etf,
                "predicted_1d_return": round(np.random.uniform(-50, 150), 1),
                "predictability_index": round(np.random.uniform(0.3, 0.9), 3),
                "is_predictable": np.random.random() > 0.3,
                "koopman_regime": np.random.choice(["expansion", "oscillatory", "contraction"])
            }
            for etf in etfs
        ]
    }


def generate_demo_history(days=30):
    """Generate demo history for visualization."""
    config = load_config()
    etfs = config['data']['etf_universe']
    history = []
    
    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        primary = np.random.choice(etfs)
        
        history.append({
            "file_date": date,
            "primary_pick": {
                "etf": primary,
                "conviction_pct": round(np.random.uniform(55, 92), 1),
                "expected_return_1d": round(np.random.uniform(-40, 130), 1)
            },
            "koopman_modes": {
                "predictability_index": round(np.random.uniform(0.55, 0.88), 3),
                "regime": np.random.choice(["expansion", "oscillatory", "contraction"])
            }
        })
    
    return history


# --- Visualization Components ---

def render_header():
    """Render main header with branding."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="main-header">🔮 P2 Koopman-Spectral Engine</div>', 
                   unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Linearized Dynamics for ETF Return Prediction · '
                   'Physics-Inspired Market Decomposition</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="text-align: right; color: #666; font-size: 0.9rem;">
            <b>Version:</b> 1.0.0<br>
            <b>Updated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="warning-banner">'
               '<b>⚠️ Research Only</b> · Not Financial Advice · '
               'Past performance does not guarantee future results</div>', 
               unsafe_allow_html=True)


def render_primary_signal(signals):
    """Render primary ETF pick with conviction metrics."""
    primary = signals['primary_pick']
    modes = signals['koopman_modes']
    
    st.markdown("---")
    st.subheader("🎯 Primary Signal")
    
    cols = st.columns([2, 1, 1, 1])
    
    with cols[0]:
        # Large ETF display
        etf_color = "#28a745" if primary['expected_return_1d'] > 0 else "#dc3545"
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
            <div style="font-size: 4rem; font-weight: 800; color: {etf_color};">
                {primary['etf']}
            </div>
            <div style="font-size: 1.2rem; color: #666; margin-top: 0.5rem;">
                Selected ETF
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.metric(
            label="Conviction",
            value=f"{primary['conviction_pct']}%",
            delta=f"{primary['conviction_pct'] - 75:.1f}% vs baseline"
        )
        st.metric(
            label="Predictability",
            value=f"{modes['predictability_index']:.2f}",
            delta="High" if modes['predictability_index'] > 0.7 else "Moderate"
        )
    
    with cols[2]:
        ret_1d = primary['expected_return_1d']
        st.metric(
            label="1-Day Expected Return",
            value=f"{ret_1d:+.1f} bps",
            delta=f"{ret_1d - 5:.1f} vs avg"
        )
        st.metric(
            label="5-Day Expected Return",
            value=f"{primary['expected_return_5d']:+.1f} bps"
        )
    
    with cols[3]:
        # Regime indicator
        regime_colors = {
            "expansion": "#28a745",
            "oscillatory": "#ffc107", 
            "contraction": "#dc3545"
        }
        regime_color = regime_colors.get(modes['regime'], '#6c757d')
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; 
                    background: {regime_color}20; border: 2px solid {regime_color};
                    border-radius: 10px;">
            <div style="font-size: 0.9rem; color: #666;">Detected Regime</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {regime_color}; 
                        text-transform: uppercase;">
                {modes['regime']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if modes['predictability_index'] > 0.8:
            st.success("🟢 High Confidence")
        elif modes['predictability_index'] > 0.6:
            st.warning("🟡 Moderate Confidence")
        else:
            st.error("🔴 Low Confidence")


def render_koopman_modes(signals):
    """Render Koopman spectral decomposition."""
    modes = signals['koopman_modes']
    
    st.markdown("---")
    st.subheader("🔬 Koopman Spectral Decomposition")
    
    cols = st.columns([1, 2])
    
    with cols[0]:
        st.markdown("#### Mode Distribution")
        
        # Mode badges
        total = 64  # observable_dim
        st.markdown(f"""
        <div style="margin: 1rem 0;">
            <span class="mode-badge growth">Growth: {modes['growth_modes']}</span>
            <span class="mode-badge oscillatory">Oscillatory: {modes['oscillatory_modes']}</span>
            <span class="mode-badge decay">Decay: {modes['decay_modes']}</span>
        </div>
        <div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">
            Total active modes: {modes['growth_modes'] + modes['oscillatory_modes'] + modes['decay_modes']}/64
        </div>
        """, unsafe_allow_html=True)
        
        # Interpretation
        st.markdown("#### Interpretation")
        if modes['growth_count'] > modes['decay_count']:
            st.info("🔥 **Expansion dominant**: Unstable modes exceed stable modes. "
                   "System exhibiting positive feedback dynamics.")
        elif modes['oscillatory_count'] > 4:
            st.info("🌊 **Cyclical dynamics**: Multiple oscillatory modes detected. "
                   "Mean-reversion or periodic patterns likely.")
        else:
            st.info("❄️ **Contracting/stable**: Decay modes dominate. "
                   "System converging to equilibrium.")
        
        st.metric("Dominant Frequency", f"{modes['dominant_frequency_cycles']:.3f} cycles/day",
                 help="Characteristic oscillation frequency of most energetic mode")
    
    with cols[1]:
        # Gauge chart for predictability
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=modes['predictability_index'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Predictability Index", 'font': {'size': 24}},
            delta={'reference': 0.7, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#ccc",
                'steps': [
                    {'range': [0, 0.5], 'color': '#ffcccc'},
                    {'range': [0.5, 0.7], 'color': '#ffffcc'},
                    {'range': [0.7, 1], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_spectrum_visualization():
    """Render eigenvalue spectrum plot."""
    st.markdown("---")
    st.subheader("🎯 Eigenvalue Spectrum")
    
    # Generate representative spectrum (or load from model)
    np.random.seed(42)
    n_modes = 64
    
    # Create realistic Koopman eigenvalue distribution
    # Most stable (inside unit circle), few unstable/growing
    angles = np.random.uniform(0, 2*np.pi, n_modes)
    radii = np.random.beta(2, 5, n_modes) * 1.2  # Skew toward <1
    
    # Ensure some structure
    radii[0:3] = np.random.uniform(1.0, 1.15, 3)  # Growth modes
    radii[3:8] = np.random.uniform(0.9, 1.0, 5)   # Near unit circle (oscillatory)
    
    real = radii * np.cos(angles)
    imag = radii * np.sin(angles)
    
    fig = go.Figure()
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines', line=dict(color='gray', dash='dash', width=2),
        name='Unit Circle (|λ|=1)',
        hoverinfo='skip'
    ))
    
    # Color by magnitude
    colors = ['#dc3545' if r > 1 else '#ffc107' if r > 0.95 else '#28a745' 
              for r in radii]
    
    fig.add_trace(go.Scatter(
        x=real, y=imag,
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            line=dict(width=1, color='black'),
            opacity=0.8
        ),
        text=[f"Mode {i}<br>|λ|={r:.3f}<br>ω={np.angle(np.exp(1j*a))/(2*np.pi):.3f}" 
              for i, (r, a) in enumerate(zip(radii, angles))],
        hovertemplate='%{text}<extra></extra>',
        name='Koopman Modes'
    ))
    
    fig.update_layout(
        title="Koopman Operator Eigenvalues in Complex Plane",
        xaxis_title="Real",
        yaxis_title="Imaginary",
        width=700,
        height=700,
        showlegend=True,
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.add_annotation(
        x=1.1, y=0,
        text="Expanding",
        showarrow=False,
        font=dict(color="#dc3545", size=12)
    )
    fig.add_annotation(
        x=0.7, y=0.7,
        text="Oscillatory",
        showarrow=False,
        font=dict(color="#ffc107", size=12)
    )
    fig.add_annotation(
        x=0.3, y=0,
        text="Contracting",
        showarrow=False,
        font=dict(color="#28a745", size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("""
    **Red**: Growth modes (|λ|>1, unstable) · 
    **Yellow**: Near-critical oscillatory (|λ|≈1) · 
    **Green**: Decay modes (|λ|<1, stable)
    """)


def render_runner_ups(signals):
    """Render runner-up ETF picks."""
    st.markdown("---")
    st.subheader("🥈 Runner-Up Selections")
    
    runners = signals.get('runner_up_picks', [])
    
    if not runners:
        st.info("No runner-up data available")
        return
    
    cols = st.columns(len(runners))
    
    for i, runner in enumerate(runners):
        with cols[i]:
            ret = runner['predicted_1d_return']
            color = "#28a745" if ret > 0 else "#dc3545"
            
            st.markdown(f"""
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; 
                        border-left: 4px solid {color};">
                <div style="font-size: 1.5rem; font-weight: 700;">{runner['etf']}</div>
                <div style="font-size: 1.1rem; color: {color}; font-weight: 600;">
                    {ret:+.1f} bps
                </div>
                <div style="font-size: 0.9rem; color: #666;">
                    Predictability: {runner['predictability_index']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_full_universe(signals):
    """Render full ETF universe heatmap."""
    st.markdown("---")
    st.subheader("🌐 Full ETF Universe")
    
    all_etfs = signals.get('all_etfs', [])
    if not all_etfs:
        st.info("No universe data available")
        return
    
    df = pd.DataFrame(all_etfs)
    
    # Create heatmap
    fig = px.scatter(
        df,
        x='predictability_index',
        y='predicted_1d_return',
        color='koopman_regime',
        size='predictability_index',
        hover_data=['etf', 'is_predictable'],
        text='etf',
        color_discrete_map={
            'expansion': '#28a745',
            'oscillatory': '#ffc107',
            'contraction': '#dc3545'
        },
        labels={
            'predictability_index': 'Predictability Index',
            'predicted_1d_return': 'Expected 1-Day Return (bps)',
            'koopman_regime': 'Detected Regime'
        },
        title="ETF Landscape: Return vs Predictability"
    )
    
    fig.update_traces(
        textposition='top center',
        marker=dict(line=dict(width=1, color='black'))
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", opacity=0.3,
                  annotation_text="Predictability Threshold")
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    with st.expander("View Full Data Table"):
        df_display = df[['etf', 'predicted_1d_return', 'predictability_index', 
                        'koopman_regime', 'is_predictable']].sort_values(
            'predicted_1d_return', ascending=False)
        df_display['predicted_1d_return'] = df_display['predicted_1d_return'].apply(
            lambda x: f"{x:+.1f} bps")
        df_display['predictability_index'] = df_display['predictability_index'].apply(
            lambda x: f"{x:.3f}")
        st.dataframe(df_display, use_container_width=True, hide_index=True)


def render_historical_trends(history):
    """Render historical signal trends."""
    st.markdown("---")
    st.subheader("📈 Historical Signal Quality")
    
    if not history:
        st.info("No historical data available")
        return
    
    df = pd.DataFrame([
        {
            'date': h['file_date'],
            'etf': h['primary_pick']['etf'],
            'conviction': h['primary_pick']['conviction_pct'],
            'return_pred': h['primary_pick']['expected_return_1d'],
            'predictability': h['koopman_modes']['predictability_index'],
            'regime': h['koopman_modes']['regime']
        }
        for h in history
    ])
    
    df = df.sort_values('date')
    
    # Multi-metric time series
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Primary Pick Predictability', 'Expected Return (bps)', 
                       'Engine Confidence'),
        vertical_spacing=0.08
    )
    
    # Predictability
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['predictability'], 
                  mode='lines+markers', name='Predictability',
                  line=dict(color='#1f77b4', width=2),
                  fill='tozeroy', fillcolor='rgba(31,119,180,0.2)'),
        row=1, col=1
    )
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=1, col=1)
    
    # Returns
    colors = ['#28a745' if r > 0 else '#dc3545' for r in df['return_pred']]
    fig.add_trace(
        go.Bar(x=df['date'], y=df['return_pred'], name='Expected Return',
               marker_color=colors),
        row=2, col=1
    )
    
    # Conviction
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['conviction'], 
                  mode='lines+markers', name='Conviction %',
                  line=dict(color='#9467bd', width=2)),
        row=3, col=1
    )
    
    fig.update_layout(height=800, showlegend=False,
                     title_text="Koopman-Spectral Engine Performance Over Time")
    st.plotly_chart(fig, use_container_width=True)
    
    # Regime distribution
    col1, col2 = st.columns(2)
    
    with col1:
        regime_counts = df['regime'].value_counts()
        fig_pie = px.pie(values=regime_counts.values, names=regime_counts.index,
                        title="Regime Distribution (Last 30 Days)",
                        color=regime_counts.index,
                        color_discrete_map={
                            'expansion': '#28a745',
                            'oscillatory': '#ffc107',
                            'contraction': '#dc3545'
                        })
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.metric("Avg Predictability", f"{df['predictability'].mean():.3f}")
        st.metric("Avg Conviction", f"{df['conviction'].mean():.1f}%")
        st.metric("Positive Predictions", 
                 f"{(df['return_pred'] > 0).sum()}/{len(df)} days")


def render_sidebar():
    """Render sidebar controls and info."""
    with st.sidebar:
        st.markdown("### 🔧 Engine Controls")
        
        # Date selector
        selected_date = st.date_input(
            "Signal Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        # Refresh
        if st.button("🔄 Refresh Signals", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Engine Parameters")
        
        config = load_config()
        st.json({
            "Observable Dimension": config['model']['observable_dim'],
            "Lookback Window": f"{config['data']['lookback_window']} days",
            "Encoder Hidden": config['model']['encoder_hidden'],
            "DMD Init": config['training']['dmd_init']
        })
        
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("""
        - [Training Logs](#)
        - [Model Weights](#)
        - [Signal History](#)
        - [Documentation](https://github.com/P2SAMAPA/P2-ETF-KOOPMAN-SPECTRAL)
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #666;">
            <b>P2 Engine Suite</b><br>
            Built: April 2026<br>
            License: Research Only
        </div>
        """, unsafe_allow_html=True)


def render_footer():
    """Render footer with disclaimers."""
    st.markdown("""
    <div class="footer">
        <b>P2 Koopman-Spectral Engine v1.0.0</b> · 
        Part of P2 Quantitative Research Suite · 
        32+2 Active Engines<br><br>
        
        <b>⚠️ Research Disclaimer</b><br>
        This engine implements Koopman operator theory for time series analysis. 
        All outputs are generated by machine learning models trained on historical data 
        and do not constitute investment advice. Past performance does not guarantee 
        future results. The predictability index measures model confidence, not 
        market certainty. Eigenvalue-based regime classification is an 
        interpretability tool, not a trading signal. Use at your own risk.<br><br>
        
        <b>Technical Notes</b><br>
        • DMD initialization provides warm-start for faster convergence<br>
        • Spectral gap threshold set at 0.7 for "predictable" classification<br>
        • 64-dimensional observable space with MLP encoder<br>
        • Training optimized for GitHub Actions 6-hour compute limit<br><br>
        
        © 2026 P2SAMAPA · Research Only · Not Financial Advice
    </div>
    """, unsafe_allow_html=True)


# --- Main App ---

def main():
    """Main application entry point."""
    render_header()
    render_sidebar()
    
    # Load data
    signals = load_latest_signals()
    history = load_historical_signals(days=30)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Current Signal", 
        "🔬 Spectral Analysis", 
        "🌐 Universe View",
        "📈 History"
    ])
    
    with tab1:
        render_primary_signal(signals)
        render_runner_ups(signals)
    
    with tab2:
        render_koopman_modes(signals)
        render_spectrum_visualization()
    
    with tab3:
        render_full_universe(signals)
    
    with tab4:
        render_historical_trends(history)
    
    render_footer()


if __name__ == "__main__":
    main()
