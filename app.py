import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import RULE_NAMES, RULE_PARAM_TYPES, PERIODS, engine_loop
from storage import save_winner, load_winners, delete_winners

st.set_page_config(
    page_title="StrategyEngine — Free Algorithmic Forex Trading Strategy Builder & Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SEO: meta tags + schema.org JSON-LD ──────────────────────────────────────
st.markdown("""
<head>
<meta name="description" content="Free algorithmic forex trading strategy builder and backtester. Build, test, and validate rule-based trading strategies using genetic algorithms and Monte Carlo simulation. Based on University of Sydney research." />
<meta name="keywords" content="algorithmic trading, forex strategy builder, backtesting, rule-based trading system, genetic algorithm trading, Monte Carlo simulation trading, quantitative trading, trading strategy tester, free backtester, forex trading tools" />
<meta name="author" content="StrategyEngine" />
<meta name="robots" content="index, follow" />
<meta property="og:type" content="website" />
<meta property="og:title" content="StrategyEngine — Free Algorithmic Forex Trading Strategy Builder" />
<meta property="og:description" content="Build and stress-test rule-based forex trading strategies. 16 technical indicators, genetic algorithm optimisation, Monte Carlo validation. Built on University of Sydney research." />
<meta property="og:image" content="https://raw.githubusercontent.com/4mmar1mtiaz/strategy-engine/master/preview.png" />
<meta name="twitter:card" content="summary_large_image" />
<meta name="twitter:title" content="StrategyEngine — Free Algorithmic Forex Trading Strategy Builder" />
<meta name="twitter:description" content="Build and stress-test rule-based forex trading strategies with 16 indicators, GA optimisation, and Monte Carlo validation." />
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "StrategyEngine",
  "applicationCategory": "FinanceApplication",
  "operatingSystem": "Web",
  "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" },
  "description": "Free algorithmic forex trading strategy builder and backtester using rule-based indicators, genetic algorithm optimisation, and Monte Carlo simulation. Built on University of Sydney research.",
  "featureList": [
    "16 technical indicator rules including MA, EMA, RSI, CCI, Bollinger Bands, Ichimoku",
    "Genetic algorithm strategy optimisation",
    "Monte Carlo stress testing",
    "Walk-forward backtesting",
    "CSV data upload for any forex pair or timeframe"
  ],
  "isBasedOn": {
    "@type": "ScholarlyArticle",
    "author": { "@type": "Organization", "name": "University of Sydney" },
    "description": "Rule-based forex trading system research — Master of Data Science capstone project"
  }
}
</script>
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is StrategyEngine?",
      "acceptedAnswer": { "@type": "Answer", "text": "StrategyEngine is a free tool for building and backtesting rule-based algorithmic forex trading strategies. It uses genetic algorithms to optimise indicator weights and Monte Carlo simulation to validate robustness." }
    },
    {
      "@type": "Question",
      "name": "What research is this based on?",
      "acceptedAnswer": { "@type": "Answer", "text": "StrategyEngine is built on a University of Sydney Master of Data Science capstone research project on autonomous rule-based forex trading systems, supervised by Dr. Matloob Khushi." }
    },
    {
      "@type": "Question",
      "name": "Is StrategyEngine free?",
      "acceptedAnswer": { "@type": "Answer", "text": "Yes, completely free. Upload your own OHLC data, select indicators, configure the engine, and let it run." }
    },
    {
      "@type": "Question",
      "name": "What technical indicators does it support?",
      "acceptedAnswer": { "@type": "Answer", "text": "16 indicators: Moving Average crossovers, EMA, DEMA, TEMA, Stochastic, Vortex, Ichimoku Cloud, RSI, CCI, Keltner Channel, Donchian Channel, and Bollinger Bands." }
    }
  ]
}
</script>
</head>
""", unsafe_allow_html=True)

# ---------- session state init ----------
if 'engine_state' not in st.session_state:
    st.session_state.engine_state = {
        'status': 'Idle — upload a CSV and press Start.',
        'phase': 'idle',
        'iteration': 0,
        'winners': [],
        'last_test': None,
        'last_mc': None,
        'rule_params': None,
        'train_bars': 0,
        'test_bars': 0,
        'total_bars': 0,
    }

if 'running' not in st.session_state:
    st.session_state.running = False

if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None

if 'thread' not in st.session_state:
    st.session_state.thread = None

# ---------- user identity (URL-based, no login) ----------
import uuid

params = st.query_params
if 'uid' not in params:
    new_uid = uuid.uuid4().hex[:10]
    st.query_params['uid'] = new_uid

user_id = st.query_params['uid']

# Load this user's saved winners into session state on first load
if 'winners_loaded' not in st.session_state:
    saved = load_winners(user_id)
    if saved:
        st.session_state.engine_state['winners'] = saved
    st.session_state.winners_loaded = True

# Rule descriptions shown as tooltips
RULE_DESCRIPTIONS = [
    "Two simple moving averages cross. Buy when fast MA crosses above slow MA.",
    "Exponential MA (reacts faster to recent prices) crosses a simple MA.",
    "Two exponential MAs cross. More responsive than pure MA crossovers.",
    "Double EMA (reduces lag further) crosses a simple MA.",
    "Two Double EMAs cross. Very low lag, sensitive to short-term moves.",
    "Triple EMA (minimal lag) crosses a simple MA. Best for fast-moving markets.",
    "Stochastic oscillator (momentum) crosses its own moving average signal line.",
    "Vortex+ (uptrend strength) crosses Vortex- (downtrend strength). Trend detection.",
    "Price relative to Ichimoku cloud spans A and B. Cloud = dynamic support/resistance.",
    "RSI crosses a single threshold. RSI < threshold = buy, > = sell.",
    "CCI crosses a single threshold. Measures how far price is from its average.",
    "RSI enters/exits overbought and oversold zones (two thresholds).",
    "CCI enters/exits overbought and oversold zones (two thresholds).",
    "Price vs Keltner Channel bands (ATR-based). Breakout = signal.",
    "Price vs Donchian Channel (rolling high/low). Breakout = signal.",
    "Price vs Bollinger Bands (std dev bands). Breakout outside bands = signal.",
]

def render_rule_param_inputs(rule_idx):
    """Render the right parameter inputs for a rule. Returns the param value."""
    ptype = RULE_PARAM_TYPES[rule_idx]
    k = f"mp_{rule_idx}"

    if ptype == 'two_periods':
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("Period 1", PERIODS, index=4, key=f"{k}_p1",
                           help="Shorter period — faster signal line.")
        p2 = c2.selectbox("Period 2", PERIODS, index=8, key=f"{k}_p2",
                           help="Longer period — slower signal line. Should be >= Period 1.")
        return (p1, p2)

    elif ptype == 'rsi_threshold':
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("RSI Period", PERIODS, index=4, key=f"{k}_p1",
                           help="Lookback bars for RSI calculation. 14 is standard.")
        p2 = c2.slider("Threshold", 0, 100, 50, key=f"{k}_p2",
                        help="RSI below this = bullish signal. Above = bearish.")
        return (p1, p2)

    elif ptype == 'cci_threshold':
        c1, c2 = st.columns(2)
        p1 = c1.selectbox("CCI Period", PERIODS, index=4, key=f"{k}_p1",
                           help="Lookback bars for CCI. 20 is standard.")
        p2 = c2.slider("Threshold", -120, 120, 0, key=f"{k}_p2",
                        help="CCI below this = bullish signal. Above = bearish.")
        return (p1, p2)

    elif ptype == 'rsi_dual_band':
        p1 = st.selectbox("RSI Period", PERIODS, index=4, key=f"{k}_p1",
                           help="Lookback bars for RSI.")
        c1, c2 = st.columns(2)
        upper = c1.slider("Overbought", 50, 100, 70, key=f"{k}_upper",
                           help="RSI above this = sell signal (overbought).")
        lower = c2.slider("Oversold", 0, 50, 30, key=f"{k}_lower",
                           help="RSI below this = buy signal (oversold).")
        return (p1, upper, lower)

    elif ptype == 'cci_dual_band':
        p1 = st.selectbox("CCI Period", PERIODS, index=4, key=f"{k}_p1",
                           help="Lookback bars for CCI.")
        c1, c2 = st.columns(2)
        upper = c1.slider("Overbought", 0, 200, 100, key=f"{k}_upper",
                           help="CCI above this = sell signal.")
        lower = c2.slider("Oversold", -200, 0, -100, key=f"{k}_lower",
                           help="CCI below this = buy signal.")
        return (p1, upper, lower)

    else:  # single_period
        p = st.selectbox("Period", PERIODS, index=4, key=f"{k}_p",
                          help="Lookback window for this channel/band indicator.")
        return p


# ---------- sidebar ----------
with st.sidebar:
    st.title("Strategy Engine")
    st.divider()

    uploaded = st.file_uploader(
        "Upload OHLC CSV",
        type=['csv'],
        help="CSV must have columns: Open, High, Low, Close. Any timeframe works — the engine will adapt.",
    )

    st.subheader("Rules")
    st.caption("Pick which indicators to include. More rules = slower training but richer strategies.")
    selected_rules = []
    for i, name in enumerate(RULE_NAMES):
        if st.checkbox(
            f"R{i+1}  {name}",
            value=True,
            key=f'rule_{i}',
            help=RULE_DESCRIPTIONS[i],
        ):
            selected_rules.append(i)

    st.divider()
    auto_train = st.toggle(
        "Auto-train rule parameters",
        value=True,
        key='auto_train',
        help=(
            "ON: brute-force searches every parameter combination on your training data "
            "to find the best settings per rule (slow but optimal). "
            "OFF: you set the parameters manually below — starts instantly."
        ),
    )

    manual_params = {}
    if not auto_train:
        if not selected_rules:
            st.caption("Select at least one rule above to configure its parameters.")
        else:
            st.caption("Set parameters for each selected rule:")
            for idx in selected_rules:
                with st.expander(f"R{idx+1} — {RULE_NAMES[idx]}"):
                    manual_params[idx] = render_rule_param_inputs(idx)

    st.divider()
    st.subheader("Data Split")
    train_pct = st.slider(
        "Training data %",
        50, 90, 80, step=5,
        help=(
            "How much of your CSV is used to train the indicator parameters. "
            "The remaining % becomes the hold-out test set — the engine never "
            "optimises on it, so performance there is a real out-of-sample result. "
            "80% train / 20% test is a good starting point."
        ),
    )

    st.subheader("Genetic Algorithm")
    ga_gen = st.number_input(
        "Generations",
        50, 1000, 200, step=50,
        help=(
            "Number of evolution cycles per GA run. More generations = more thorough "
            "search but slower. 200 is a solid default; try 400+ if you have time."
        ),
    )
    ga_pop = st.number_input(
        "Population size",
        4, 30, 10, step=2,
        help=(
            "How many candidate weight combinations the GA maintains at once. "
            "Bigger population = more diversity, slower per generation. 8-12 is typical."
        ),
    )
    ga_parents = st.number_input(
        "Parents mating",
        2, int(ga_pop) - 1, min(6, int(ga_pop) - 1), step=1,
        help=(
            "How many top individuals are selected each generation to breed the next. "
            "Must be less than population size. Higher = more elitism, less exploration."
        ),
    )

    st.subheader("Pre-MC Filters")
    st.caption("Quick checks before running the expensive Monte Carlo. Saves time by skipping bad strategies early.")
    min_return = st.number_input(
        "Min return % on test set",
        0.0, 100.0, 5.0, step=1.0,
        help=(
            "A strategy must earn at least this much on the hold-out test set "
            "before Monte Carlo is run. Raise this to only MC-test strong strategies. "
            "Lower it to be more permissive."
        ),
    )
    min_sharpe = st.number_input(
        "Min Sharpe ratio on test set",
        0.0, 20.0, 0.5, step=0.1,
        help=(
            "Sharpe ratio = return / volatility (annualised). Filters out strategies "
            "that made money erratically. A ratio above 1.0 is good; above 2.0 is strong. "
            "0.5 is a loose filter to start."
        ),
    )

    st.subheader("Monte Carlo")
    st.caption("Stress-tests each strategy by running it on hundreds of slightly altered versions of your data.")
    num_mc = st.slider(
        "Simulations",
        10, 300, 50, step=10,
        help=(
            "Each simulation adds random noise to your price data and re-runs the strategy. "
            "More sims = more confidence but slower. 50 is fast; 200+ for higher confidence."
        ),
    )
    mc_noise = st.number_input(
        "Price noise std",
        0.0001, 0.01, 0.0001, format="%.4f",
        help=(
            "Standard deviation of the log-normal noise added to each OHLC bar. "
            "0.0001 = ~0.01% per bar (very subtle). Increase to 0.001 to stress-test "
            "against bigger data imperfections or slippage."
        ),
    )

    st.divider()
    st.subheader("Winner Criteria")
    st.caption("A strategy must pass ALL of these to be saved as a winner.")
    winner_min_return = st.number_input(
        "Min return %",
        0.0, 500.0, 10.0, step=5.0,
        help=(
            "Minimum total return % on the test set for a strategy to be saved. "
            "Tune this to your data — a 3-month dataset might target 15-20%, "
            "a 1-year dataset might target 50-100%."
        ),
    )
    winner_min_sharpe = st.number_input(
        "Min Sharpe ratio",
        0.0, 20.0, 1.0, step=0.1,
        help=(
            "Minimum annualised Sharpe ratio. Filters out strategies that made money "
            "erratically. 1.0 = decent, 2.0 = strong, 5.0+ = exceptional."
        ),
    )
    winner_max_dd = st.number_input(
        "Max drawdown % (absolute)",
        0.0, 100.0, 10.0, step=1.0,
        help=(
            "Maximum allowed peak-to-trough loss on the test set. "
            "Enter as a positive number — e.g. 10 means the strategy can drop at most -10% "
            "from its peak before it's disqualified."
        ),
    )
    mc_thresh = st.slider(
        "Min MC pass rate %",
        50, 100, 60,
        help=(
            "Minimum % of Monte Carlo simulations that must be profitable. "
            "60% = profitable in 6 out of 10 noisy scenarios. "
            "Raise to 75-80% for higher confidence winners."
        ),
    )

    st.divider()
    st.subheader("Your save code")
    st.code(user_id, language=None)
    st.caption(
        "Bookmark your URL or copy this code — your winners are always saved against it. "
        "Anyone with your code can see your strategies, so keep it private."
    )
    if st.button("Clear all my saved winners", type="secondary",
                 help="Permanently deletes all winners saved under your code."):
        delete_winners(user_id)
        st.session_state.engine_state['winners'] = []
        st.session_state.winners_loaded = False
        st.rerun()

    st.divider()
    col_start, col_stop = st.columns(2)
    start_btn = col_start.button(
        "Start",
        disabled=st.session_state.running or uploaded is None or len(selected_rules) == 0,
        use_container_width=True,
        type="primary",
    )
    stop_btn = col_stop.button(
        "Stop",
        disabled=not st.session_state.running,
        use_container_width=True,
    )

# ---------- start ----------
if start_btn:
    data = pd.read_csv(uploaded)
    required = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required if c not in data.columns]
    if missing:
        st.sidebar.error(f"CSV missing columns: {missing}")
    else:
        data = data[required].copy()

        config = {
            'data': data,
            'selected_indices': selected_rules,
            'train_pct': int(train_pct),
            'ga_generations': int(ga_gen),
            'ga_pop': int(ga_pop),
            'ga_parents': int(ga_parents),
            'num_mc_sims': int(num_mc),
            'mc_noise_std': float(mc_noise),
            'mc_pass_threshold': float(mc_thresh),
            'min_return': float(min_return),
            'min_sharpe': float(min_sharpe),
            'winner_min_return': float(winner_min_return),
            'winner_min_sharpe': float(winner_min_sharpe),
            'winner_max_dd': float(winner_max_dd),
            'manual_params': manual_params if not auto_train else None,
            'user_id': user_id,
        }

        stop_event = threading.Event()
        state = st.session_state.engine_state
        state.update({
            'status': 'Starting...',
            'phase': 'training',
            'iteration': 0,
            'winners': [],
            'last_test': None,
            'last_mc': None,
            'rule_params': None,
        })

        thread = threading.Thread(
            target=engine_loop,
            args=(config, state, stop_event),
            daemon=True,
        )
        thread.start()

        st.session_state.running = True
        st.session_state.stop_event = stop_event
        st.session_state.thread = thread
        st.rerun()

# ---------- stop ----------
if stop_btn:
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    st.session_state.running = False
    st.rerun()

# ---------- landing page ----------
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0d1b2a 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 48px 40px 40px 40px;
    margin-bottom: 8px;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0 0 12px 0;
    color: #ffffff;
    line-height: 1.2;
}
.hero h1 span { color: #00d4aa; }
.hero p.sub {
    font-size: 1.15rem;
    color: #a0aec0;
    max-width: 680px;
    margin: 0 0 28px 0;
    line-height: 1.6;
}
.uni-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 212, 170, 0.1);
    border: 1px solid rgba(0, 212, 170, 0.3);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 0.82rem;
    color: #00d4aa;
    margin-bottom: 28px;
    font-weight: 500;
}
.feature-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 4px;
}
.pill {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 5px 14px;
    font-size: 0.82rem;
    color: #cbd5e0;
}
.stats-row {
    display: flex;
    gap: 32px;
    margin-top: 32px;
    padding-top: 28px;
    border-top: 1px solid rgba(255,255,255,0.08);
}
.stat { display: flex; flex-direction: column; }
.stat-num { font-size: 1.7rem; font-weight: 700; color: #fff; }
.stat-label { font-size: 0.78rem; color: #718096; margin-top: 2px; }
</style>

<div class="hero">
  <h1>Your trading data goes in.<br/><span>Winning strategies come out.</span></h1>
  <p class="sub">
    Upload a spreadsheet of price data for any currency pair. The engine automatically
    figures out the best settings for 16 popular trading indicators — things like moving averages,
    RSI, and Bollinger Bands — then mixes and matches them to find combinations that would have
    made money on your data. Every strategy that looks promising gets stress-tested under hundreds
    of simulated market conditions before it's declared a winner. No coding required.
  </p>

  <div class="feature-pills">
    <span class="pill">Step 1 — Upload your price data (CSV)</span>
    <span class="pill">Step 2 — Pick which indicators to use</span>
    <span class="pill">Step 3 — Hit Start and let it run</span>
    <span class="pill">Step 4 — Download your winning strategies</span>
  </div>

  <div class="stats-row">
    <div class="stat">
      <span class="stat-num">16</span>
      <span class="stat-label">Indicator rules</span>
    </div>
    <div class="stat">
      <span class="stat-num">∞</span>
      <span class="stat-label">Strategy combinations</span>
    </div>
    <div class="stat">
      <span class="stat-num">100s</span>
      <span class="stat-label">Stress-test simulations per strategy</span>
    </div>
    <div class="stat">
      <span class="stat-num">Free</span>
      <span class="stat-label">Always</span>
    </div>
  </div>

  <div style="margin-top:32px; padding-top:24px; border-top:1px solid rgba(255,255,255,0.08);">
    <div class="uni-badge" style="margin-bottom:14px;">
      🎓 Built on University of Sydney research &nbsp;·&nbsp; Master of Data Science capstone project
    </div>
    <p style="font-size:0.85rem; color:#718096; max-width:700px; line-height:1.7; margin:0 0 14px 0;">
      This tool is built on top of a research project from the University of Sydney's Master of Data Science programme.
      We've extended it with a full UI, automatic parameter training, and Monte Carlo stress testing to make it accessible
      to everyone — not just researchers. It is a <strong style="color:#a0aec0">free educational resource</strong>.
    </p>
    <p style="font-size:0.82rem; color:#4a5568; max-width:700px; line-height:1.7; margin:0;">
      <strong style="color:#4a5568">Disclaimer:</strong> Markets are unpredictable and past performance never guarantees future results.
      A strategy that worked on historical data can fail tomorrow — that's just how markets work. This tool is for
      learning and research only. It is <strong style="color:#4a5568">not financial advice</strong>. Never risk money you can't afford to lose.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.save-banner {{
    background: rgba(0,212,170,0.07);
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: 10px;
    padding: 16px 22px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
}}
.save-banner .icon {{ font-size: 1.5rem; }}
.save-banner .text {{ flex: 1; }}
.save-banner .text strong {{ color: #00d4aa; font-size: 0.95rem; }}
.save-banner .text p {{ margin: 4px 0 0 0; font-size: 0.82rem; color: #a0aec0; line-height: 1.5; }}
.save-banner .code-box {{
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 6px;
    padding: 6px 14px;
    font-family: monospace;
    font-size: 1rem;
    color: #00d4aa;
    letter-spacing: 0.05em;
}}
</style>
<div class="save-banner">
  <div class="icon">💾</div>
  <div class="text">
    <strong>Your progress is saved automatically</strong>
    <p>
      Your save code is <b style="color:#00d4aa; font-family:monospace">{user_id}</b> —
      bookmark this page or copy the code and share it with no one.
      Every strategy you find is saved against this code and will be here when you come back.<br>
      <span style="color:#718096">Note: saved data may reset if the app is updated or redeployed.</span>
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.how-strip {
    display: flex;
    gap: 0;
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 24px;
}
.how-step {
    flex: 1;
    padding: 18px 20px;
    border-right: 1px solid #2d3748;
    font-size: 0.83rem;
    color: #a0aec0;
    line-height: 1.5;
}
.how-step:last-child { border-right: none; }
.how-step strong { display: block; color: #e2e8f0; font-size: 0.88rem; margin-bottom: 4px; }
.step-num { color: #00d4aa; font-weight: 700; font-size: 0.75rem; margin-bottom: 6px; display: block; }
</style>
<div class="how-strip">
  <div class="how-step"><span class="step-num">STEP 1</span><strong>Upload data</strong>Any OHLC CSV — any pair, any timeframe</div>
  <div class="how-step"><span class="step-num">STEP 2</span><strong>Pick indicators</strong>Choose from 16 technical rules in the sidebar</div>
  <div class="how-step"><span class="step-num">STEP 3</span><strong>Set your bar</strong>Define what return, Sharpe &amp; drawdown qualifies as a winner</div>
  <div class="how-step"><span class="step-num">STEP 4</span><strong>Press Start</strong>Engine trains, runs GA, stress-tests with Monte Carlo</div>
  <div class="how-step"><span class="step-num">STEP 5</span><strong>Collect winners</strong>Strategies that pass everything get saved to the Winners tab</div>
</div>
""", unsafe_allow_html=True)

# ---------- main area ----------
state = st.session_state.engine_state

# Status bar
phase = state.get('phase', 'idle')
if phase == 'training':
    st.info(f"**Training rules...** {state['status']}")
elif phase == 'running':
    st.success(f"**Running** — {state['status']}")
elif phase == 'stopped':
    st.warning(f"**Stopped** — {state['status']}")
else:
    st.info(state['status'])

# Top metrics row
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "Iterations",
    state.get('iteration', 0),
    help="Total GA runs completed. Each iteration is one independent attempt to find a winning strategy.",
)
m2.metric(
    "Winners found",
    len(state.get('winners', [])),
    help="Strategies that passed both the pre-MC filter and the Monte Carlo pass threshold.",
)
m3.metric(
    "Train bars",
    state.get('train_bars', '—'),
    help="Number of OHLC bars used to train indicator parameters and the GA.",
)
m4.metric(
    "Test bars",
    state.get('test_bars', '—'),
    help="Hold-out bars the engine never optimises on. Performance here is the real out-of-sample result.",
)

last = state.get('last_test')
if last:
    m5.metric(
        "Last MC pass %",
        f"{state.get('last_mc', '—')}%",
        help="% of Monte Carlo simulations the last filtered strategy was profitable in.",
    )
else:
    m5.metric("Last MC pass %", "—")

st.divider()

# Tabs
tab_live, tab_winners, tab_rules, tab_guide = st.tabs(["Live", "Winners", "Rule Params", "Guide"])

# --- Live tab ---
with tab_live:
    if last:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Last return %",
            f"{last['return']:.2f}%",
            help="Total return of the last GA strategy on the test set.",
        )
        c2.metric(
            "Last Sharpe",
            f"{last['sharpe']:.2f}",
            help="Annualised Sharpe ratio. Above 1.0 = good risk-adjusted return.",
        )
        c3.metric(
            "Last max DD %",
            f"{last['dd']:.3f}%",
            help="Largest peak-to-trough loss during the test period. Closer to 0 = better.",
        )
    else:
        st.caption("No iterations completed yet.")

    winners = state.get('winners', [])
    if len(winners) > 1:
        st.subheader("Return % — winners over time")
        df_chart = pd.DataFrame(winners)
        if 'iteration' in df_chart.columns:
            chart_data = df_chart[['iteration', 'return_pct']].set_index('iteration')
        else:
            chart_data = df_chart[['return_pct']].reset_index()
            chart_data.columns = ['iteration', 'return_pct']
            chart_data = chart_data.set_index('iteration')
        st.line_chart(chart_data)

# --- Winners tab ---
with tab_winners:
    winners = state.get('winners', [])

    if not winners:
        st.info("No winners yet. Run the engine — strategies that pass all your criteria will appear here.")
    else:
        df_w = pd.DataFrame(winners)
        display_cols = ['timestamp', 'iteration', 'return_pct', 'max_drawdown_pct',
                        'sharpe_ratio', 'mc_pass_rate', 'strategy_hash']
        display_cols = [c for c in display_cols if c in df_w.columns]

        st.dataframe(
            df_w[display_cols].rename(columns={
                'return_pct': 'Return %',
                'max_drawdown_pct': 'Max DD %',
                'sharpe_ratio': 'Sharpe',
                'mc_pass_rate': 'MC Pass %',
                'strategy_hash': 'Hash',
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ── Bulk download ────────────────────────────────────────────
        csv_bytes = df_w.to_csv(index=False).encode()
        st.download_button(
            "Download all winners (CSV)",
            csv_bytes,
            file_name="winners_export.csv",
            mime="text/csv",
            help="Full table with all metrics and raw weights.",
        )

        st.divider()

        # ── Per-strategy download ────────────────────────────────────
        st.subheader("Download individual strategy config")
        st.caption("Pick a winner and download its full parameters as a JSON file you can plug into any script or notebook.")

        winner_labels = [
            f"#{i+1}  |  {w.get('timestamp','')[:16]}  |  Return: {w.get('return_pct','?')}%  |  Sharpe: {w.get('sharpe_ratio','?')}  |  MC: {w.get('mc_pass_rate','?')}%"
            for i, w in enumerate(winners)
        ]
        selected_idx = st.selectbox("Select winner", range(len(winner_labels)),
                                     format_func=lambda i: winner_labels[i],
                                     help="Choose a strategy to inspect or download.")

        selected_winner = winners[selected_idx]

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Return %",   f"{selected_winner.get('return_pct', '—')}%")
        sc2.metric("Max DD %",   f"{selected_winner.get('max_drawdown_pct', '—')}%")
        sc3.metric("Sharpe",     f"{selected_winner.get('sharpe_ratio', '—')}")
        sc4.metric("MC Pass %",  f"{selected_winner.get('mc_pass_rate', '—')}%")

        if 'rules_used' in selected_winner:
            st.caption(f"Rules: {selected_winner['rules_used']}")

        # Build clean JSON config for download
        import json
        strategy_config = {
            "metadata": {
                "timestamp":       selected_winner.get("timestamp", ""),
                "strategy_hash":   selected_winner.get("strategy_hash", ""),
                "return_pct":      selected_winner.get("return_pct"),
                "max_drawdown_pct":selected_winner.get("max_drawdown_pct"),
                "sharpe_ratio":    selected_winner.get("sharpe_ratio"),
                "mc_pass_rate":    selected_winner.get("mc_pass_rate"),
                "rules_used":      selected_winner.get("rules_used", ""),
            },
            "rule_params": selected_winner.get("all_rules", ""),
            "weights":     selected_winner.get("all_weights", ""),
            "usage": {
                "note": "Pass rule_params and weights into getTradingRuleFeatures() and evaluate() from tradingrule.py / engine.py",
                "example": (
                    "import ast, numpy as np\n"
                    "from tradingrule import getTradingRuleFeatures\n"
                    "rule_params = ast.literal_eval(config['rule_params'])\n"
                    "weights = np.array(ast.literal_eval(config['weights']))\n"
                    "features = getTradingRuleFeatures(your_df, rule_params)\n"
                    "position = (features.values[:, 1:] @ weights).flatten()"
                ),
            },
        }

        json_bytes = json.dumps(strategy_config, indent=2).encode()
        st.download_button(
            "Download strategy config (JSON)",
            json_bytes,
            file_name=f"strategy_{selected_winner.get('strategy_hash', selected_idx)}.json",
            mime="application/json",
            help="Contains rule params, weights, and a usage snippet for plugging into your own scripts.",
        )

        with st.expander("Preview config"):
            st.json(strategy_config)

# --- Rule params tab ---
with tab_rules:
    rp = state.get('rule_params')
    if rp:
        st.caption(
            "Optimal indicator parameters found by brute-force search on your training data. "
            "These are fixed for the entire session — the GA only optimises the weights."
        )
        for rule_name, param_val in rp.items():
            st.text(f"{rule_name:20s}  {param_val}")
    else:
        st.info("Rule params will appear here after the training phase completes.")

# --- Guide tab ---
with tab_guide:

    # ── Quick reference ──────────────────────────────────────────────
    st.subheader("Quick reference")
    st.info(
        "**Lost? Start here.**  \n"
        "1. Upload CSV → 2. Pick rules → 3. Set winner criteria → 4. Press Start → 5. Wait for winners  \n"
        "Everything else is optional tuning."
    )

    qc1, qc2, qc3 = st.columns(3)
    qc1.success(
        "**First time setup**  \n"
        "- All rules ON  \n"
        "- Auto-train ON  \n"
        "- Split 80 / 20  \n"
        "- GA: 200 gen, pop 10  \n"
        "- MC: 50 sims  \n"
        "- Winner: 10% return, 1.0 SR"
    )
    qc2.warning(
        "**Want more winners?**  \n"
        "Lower the winner thresholds:  \n"
        "- Return % → 5%  \n"
        "- Sharpe → 0.5  \n"
        "- Max DD → 20%  \n"
        "- MC pass rate → 55%"
    )
    qc3.error(
        "**Want fewer but stronger?**  \n"
        "Raise winner thresholds:  \n"
        "- Return % → 20%+  \n"
        "- Sharpe → 2.0+  \n"
        "- Max DD → 5%  \n"
        "- MC pass rate → 75%+"
    )

    st.divider()

    # ── Full pipeline ─────────────────────────────────────────────────
    st.subheader("The full pipeline — what actually runs")

    with st.expander("Step 1 — Upload your data", expanded=False):
        st.markdown("""
**What:** Any CSV with `Open`, `High`, `Low`, `Close` columns.

**What happens to it:**
- Split into **training set** (used to find indicator params + GA weights) and **test set** (hold-out — never touched during optimisation)
- Training % is set by the slider. Default 80% train / 20% test.

**Key point:** The test set is the only honest measure of performance. If a strategy looks good there, it wasn't just fitted to known data.

**CSV format:**
```
Open,High,Low,Close
1.2345,1.2360,1.2330,1.2350
...
```
Timeframe doesn't matter — 1min, 5min, daily all work. The engine adapts.
        """)

    with st.expander("Step 2 — Rule parameter training (slow, runs once)", expanded=False):
        st.markdown("""
**What:** For each selected indicator rule, the engine tries every combination of parameters on your training data and picks the one with the highest return.

**Example for R1 (MA × MA):**
```
Try period pairs: (1,1), (1,3), (1,5) ... (50,61) — 91 combinations
→ Pick the pair that produced the best return on training data
```

**Why it's slow:** Some rules have hundreds of combinations. 16 rules × hundreds of combos = a few minutes.

**Skip it:** Toggle off "Auto-train rule parameters" in the sidebar and set them manually — starts instantly.

**After training:** Check the **Rule Params** tab to see what the engine found. You can copy those values into manual mode for future runs on similar data.
        """)

    with st.expander("Step 3 — Genetic Algorithm loop (runs forever)", expanded=False):
        st.markdown("""
**What:** The GA finds the best *combination of weights* — how much to trust each rule's signal.

**Each iteration:**
1. Randomly creates a population of weight sets
2. Calculates fitness for each (GAMSSR ratio — explained below)
3. Keeps the best, breeds them together, mutates slightly
4. Repeats for N generations
5. Returns the best weight set found

**GAMSSR fitness function:**
`mean(returns) / std(returns) / sum(all_losses)`
= Sharpe-like, but **heavily penalises losses**. A strategy that makes money smoothly scores much higher than one that makes the same return with big drawdowns.

**Key point:** Rule params are **fixed**. The GA only moves weights. You get a different set of weights every iteration because the GA starts from a random population each time.

**Tuning:**
- More generations → deeper search, slower
- Bigger population → more diversity, slower per generation
- More parents mating → more elitism, less exploration
        """)

    with st.expander("Step 4 — Pre-MC filter (fast gate)", expanded=False):
        st.markdown("""
**What:** A quick check before running the expensive Monte Carlo. If the strategy doesn't clear these, it's thrown away immediately.

**Two checks:**
- Return % on test set ≥ your minimum
- Sharpe ratio on test set ≥ your minimum

**Why have this?** Monte Carlo takes time. Without this gate, you'd run MC on every mediocre strategy. Set these just above "obviously bad" — they're not your real winner criteria, just a speed filter.

**Recommended starting values:** Return 5%, Sharpe 0.5
        """)

    with st.expander("Step 5 — Monte Carlo stress test", expanded=False):
        st.markdown("""
**What:** Runs the strategy N times on slightly different versions of your data to see if it still works when the data isn't perfect.

**How each simulation works:**
1. Add tiny random noise to every OHLC bar (log-normal, controlled by noise std)
2. Re-run the strategy on the noisy data
3. Record whether it was profitable

**Pass rate** = % of simulations that were profitable

**Why this matters:** A strategy that only works on the exact data it was tested on is probably overfit. If it still makes money when prices are nudged slightly, it's more likely to work on future data.

**Noise std guide:**
- `0.0001` = ~0.01% per bar — very subtle, tests minor data differences
- `0.001` = ~0.1% per bar — moderate, simulates slippage / bid-ask spread
- `0.005` = ~0.5% per bar — aggressive stress test

**Sims guide:**
- 50 sims = fast, rough confidence
- 100 sims = good balance
- 200+ sims = high confidence, slower
        """)

    with st.expander("Step 6 — Winner criteria (your real standards)", expanded=False):
        st.markdown("""
**What:** The final gate. A strategy must pass ALL FOUR to be saved.

| Criterion | What it means | Tune based on |
|---|---|---|
| Min return % | Total profit on test set | Length of your data |
| Min Sharpe | Return vs volatility | How smooth you want it |
| Max drawdown % | Biggest loss from peak | Your risk tolerance |
| Min MC pass rate % | Robustness across noisy scenarios | How confident you want to be |

**All four must pass.** The status bar tells you exactly which one failed — e.g. `"Failed: return 3.2% < 10%"` — so you know what to adjust.

**Tune return % to your data length:**
- 1 month of 5-min data → target 5–15%
- 3 months → target 15–30%
- 1 year → target 50–100%+
        """)

    st.divider()

    # ── Indicator rules ───────────────────────────────────────────────
    st.subheader("Indicator rules — what each one does")

    RULE_TAGS = [
        "trend", "trend", "trend", "trend", "trend", "trend",
        "momentum", "trend", "trend",
        "momentum", "momentum", "momentum", "momentum",
        "volatility", "volatility", "volatility",
    ]
    TAG_COLORS = {"trend": "🔵", "momentum": "🟠", "volatility": "🟣"}

    tag_legend_col1, tag_legend_col2, tag_legend_col3 = st.columns(3)
    tag_legend_col1.caption("🔵 Trend — follow direction")
    tag_legend_col2.caption("🟠 Momentum — overbought/oversold")
    tag_legend_col3.caption("🟣 Volatility — channel breakouts")

    for i, (name, desc) in enumerate(zip(RULE_NAMES, RULE_DESCRIPTIONS)):
        tag = RULE_TAGS[i]
        icon = TAG_COLORS[tag]
        with st.expander(f"{icon} R{i+1} — {name}"):
            st.markdown(f"**What it does:** {desc}")
            ptype = RULE_PARAM_TYPES[i]
            if ptype == 'two_periods':
                st.caption("Parameters: Period 1 (fast line), Period 2 (slow line). Signal fires when they cross.")
            elif ptype in ('rsi_threshold', 'cci_threshold'):
                st.caption("Parameters: Indicator period, Threshold level. Signal fires when indicator crosses threshold.")
            elif ptype in ('rsi_dual_band', 'cci_dual_band'):
                st.caption("Parameters: Period, Overbought level (sell signal), Oversold level (buy signal).")
            else:
                st.caption("Parameters: Single period for the channel/band width.")

    st.divider()

    # ── Settings cheat sheet ──────────────────────────────────────────
    st.subheader("Settings cheat sheet")

    with st.expander("Speed vs quality tradeoffs", expanded=False):
        st.markdown("""
| Setting | Faster / more iterations | Slower / deeper search |
|---|---|---|
| GA generations | 50–100 | 300–500 |
| GA population | 4–6 | 16–20 |
| MC simulations | 20–30 | 150–200 |
| Rules selected | 4–6 rules | All 16 |
| Auto-train | OFF (manual params) | ON |

**Tip:** Start fast to confirm the setup works, then slow down for the overnight run.
        """)

    with st.expander("Nothing is being saved as a winner — what to check", expanded=False):
        st.markdown("""
Watch the **status bar** — it tells you exactly what failed.

**Common causes:**

`Failed: return 3.2% < 10%`
→ Lower "Min return %" in Winner Criteria, or use more data

`Failed: Sharpe 0.4 < 1.0`
→ Lower "Min Sharpe" — your data may have inherently noisy returns

`Failed: DD 15% > 10%`
→ Raise "Max drawdown %" — the strategy is volatile but still profitable

`Failed: MC 45% < 60%`
→ Lower "Min MC pass rate" or reduce noise std — strategy may be slightly sensitive to noise

`No iterations completing at all`
→ Pre-MC filters are too tight. Lower "Min return %" and "Min Sharpe" in the Pre-MC Filters section.
        """)

    with st.expander("Auto-train vs manual parameters — when to use each", expanded=False):
        st.markdown("""
**Use Auto-train when:**
- First time running on a new dataset
- You want the system to figure out what works
- You have time (coffee run)

**Use Manual when:**
- You've already run auto-train and noted the params from the Rule Params tab
- You want to test a specific hypothesis ("what if RSI uses 14 / 70 / 30?")
- You want to start immediately without waiting

**Pro workflow:**
1. Run auto-train once → check Rule Params tab → note the values
2. Next session: switch to manual, enter those values → skip straight to GA
        """)

    with st.expander("Understanding the metrics", expanded=False):
        st.markdown("""
**Return %** — total profit over the test period as a percentage. Simple.

**Sharpe ratio** — return divided by volatility, annualised. Tells you if the returns were smooth or chaotic.
- Below 0.5 → bad
- 0.5–1.0 → acceptable
- 1.0–2.0 → good
- 2.0–5.0 → strong
- 5.0+ → exceptional (verify it's not overfit)

**Max drawdown %** — the biggest drop from a peak before recovering. If your strategy hit +20% then fell to +10% before recovering, that's a -10% drawdown.
- Below 5% → very tight
- 5–15% → normal
- Above 20% → risky

**MC pass rate %** — out of N noisy simulations, how many were profitable.
- Below 55% → basically random
- 60–70% → decent robustness
- 75%+ → confident it's not a fluke
        """)

    st.divider()
    st.caption("Tip: the status bar at the top always shows exactly what the engine is doing and why strategies pass or fail.")

# ---------- footer ----------
st.markdown("""
<style>
.footer {
    margin-top: 60px;
    padding: 24px 0 12px 0;
    border-top: 1px solid #2d3748;
    text-align: center;
    color: #4a5568;
    font-size: 0.82rem;
}
.footer span { color: #718096; }
</style>
<div class="footer">
    Created by <strong style="color:#a0aec0">Ammar</strong>
    &nbsp;·&nbsp;
    <span>Built on University of Sydney research</span>
    &nbsp;·&nbsp;
    <span>Free forever</span>
</div>
""", unsafe_allow_html=True)

# ---------- auto-refresh while running ----------
if st.session_state.running:
    time.sleep(2)
    st.rerun()
