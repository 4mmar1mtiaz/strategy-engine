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

from engine import RULE_NAMES, engine_loop

st.set_page_config(
    page_title="Strategy Engine",
    page_icon="📈",
    layout="wide",
)

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
        chart_data = pd.DataFrame(winners)[['iteration', 'return_pct']].set_index('iteration')
        st.line_chart(chart_data)

# --- Winners tab ---
with tab_winners:
    winners = state.get('winners', [])
    if not winners:
        st.info("No winners yet. The engine needs to find strategies that pass both the pre-MC filter and Monte Carlo validation.")
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

        csv_bytes = df_w.to_csv(index=False).encode()
        st.download_button(
            "Download winners CSV",
            csv_bytes,
            file_name="winners_export.csv",
            mime="text/csv",
        )

        st.subheader("Best winner")
        best = df_w.loc[df_w['return_pct'].idxmax()]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Return %", f"{best['return_pct']:.2f}%")
        b2.metric("Max DD %", f"{best['max_drawdown_pct']:.3f}%")
        b3.metric("Sharpe", f"{best['sharpe_ratio']:.2f}")
        b4.metric("MC Pass %", f"{best['mc_pass_rate']:.1f}%")
        if 'rules_used' in best:
            st.caption(f"Rules used: {best['rules_used']}")

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
    st.subheader("How it works")
    st.markdown("""
    **Step 1 — Upload your data**
    Upload any CSV with `Open`, `High`, `Low`, `Close` columns. The engine splits it into
    a training set and a hold-out test set based on your configured split %.

    **Step 2 — Rule training (once per session)**
    For each selected indicator rule, the engine brute-forces through hundreds of parameter
    combinations (e.g. MA periods, RSI thresholds) and picks the best-performing one on the
    training data. This is the slow part — expect a few minutes depending on how many rules you selected.

    **Step 3 — GA loop (runs forever)**
    Once rules are trained, the Genetic Algorithm loops continuously. Each iteration:
    - Randomly initialises a population of weight combinations
    - Evolves them over N generations, selecting and breeding the best performers
    - The fitness function is the GAMSSR ratio — a Sharpe-like metric that penalises losses hard

    **Step 4 — Pre-MC filter**
    The winning weights are applied to the test set. If the strategy doesn't meet your
    minimum return % and Sharpe ratio, it's discarded and the loop continues.

    **Step 5 — Monte Carlo validation**
    For strategies that pass the pre-MC filter, Monte Carlo runs N simulations.
    Each sim adds random price noise to the test data and re-runs the strategy.

    **Step 6 — Winner criteria**
    After MC, the strategy must also meet your Winner Criteria thresholds:
    min return %, min Sharpe, max drawdown %, and min MC pass rate.
    All four must be satisfied — if any one fails, the strategy is discarded.

    **Step 6 — Winners**
    Winning strategies are shown in the Winners tab and appended to `winners.csv`.
    Each winner includes its weights, rules used, return, Sharpe, drawdown, and MC pass rate.
    """)

    st.subheader("Indicator rules")
    for i, (name, desc) in enumerate(zip(RULE_NAMES, RULE_DESCRIPTIONS)):
        st.markdown(f"**R{i+1} — {name}:** {desc}")

    st.subheader("Tips")
    st.markdown("""
    - **Start with all rules selected** to let the GA pick the best combination via weights
    - **Tighten the pre-MC filters** (raise min return / Sharpe) to only MC-test promising strategies — saves time
    - **Raise MC pass threshold to 75-80%** for higher confidence winners
    - **Lower GA generations to 100** if you want faster iterations; raise to 500 for deeper search
    - The engine keeps running until you press Stop — winners accumulate in the table and CSV
    """)

# ---------- auto-refresh while running ----------
if st.session_state.running:
    time.sleep(2)
    st.rerun()
