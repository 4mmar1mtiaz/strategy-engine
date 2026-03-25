import matplotlib
matplotlib.use('Agg')  # Must be before any other matplotlib import

import numpy as np
import pandas as pd
import time
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ga as ga_module
from tradingrule import (
    Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8,
    Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15, Rule16,
)

ALL_RULES = [
    Rule1, Rule2, Rule3, Rule4, Rule5, Rule6, Rule7, Rule8,
    Rule9, Rule10, Rule11, Rule12, Rule13, Rule14, Rule15, Rule16,
]

RULE_NAMES = [
    "MA x MA",
    "EMA x MA",
    "EMA x EMA",
    "DEMA x MA",
    "DEMA x DEMA",
    "TEMA x MA",
    "Stochastic",
    "Vortex",
    "Ichimoku",
    "RSI threshold",
    "CCI threshold",
    "RSI dual band",
    "CCI dual band",
    "Keltner Channel",
    "Donchian Channel",
    "Bollinger Bands",
]

PERIODS = [1, 3, 5, 7, 11, 15, 19, 23, 27, 35, 41, 50, 61]
RSI_LIMITS = list(range(0, 101, 5))
CCI_LIMITS = list(range(-120, 121, 20))

TYPE1    = {0, 1, 2, 3, 4, 5, 6, 7, 8}
TYPE2_RSI = {9}
TYPE2_CCI = {10}
TYPE3_RSI = {11}
TYPE3_CCI = {12}
TYPE4    = {13, 14, 15}

# Exported so the UI can build the right inputs per rule type
RULE_PARAM_TYPES = {
    **{i: 'two_periods'   for i in TYPE1},
    9:  'rsi_threshold',
    10: 'cci_threshold',
    11: 'rsi_dual_band',
    12: 'cci_dual_band',
    **{i: 'single_period' for i in TYPE4},
}


def train_selected_rules(df, selected_indices, state):
    """Brute-force train only the selected rules. Returns {idx: best_params}."""
    OHLC = [df.Open, df.High, df.Low, df.Close]
    params = {}

    for i, idx in enumerate(selected_indices):
        state['status'] = (
            f"Training rules ({i + 1}/{len(selected_indices)}): {RULE_NAMES[idx]}..."
        )
        rule = ALL_RULES[idx]

        if idx in TYPE1:
            best, best_p = -1, (PERIODS[0], PERIODS[0])
            for a in range(len(PERIODS)):
                for b in range(a, len(PERIODS)):
                    p = (PERIODS[a], PERIODS[b])
                    s = rule(p, OHLC)[0]
                    if s > best:
                        best, best_p = s, p
            params[idx] = best_p

        elif idx in TYPE2_RSI:
            best, best_p = -1, (PERIODS[0], RSI_LIMITS[0])
            for period in PERIODS:
                for lim in RSI_LIMITS:
                    p = (period, lim)
                    s = rule(p, OHLC)[0]
                    if s > best:
                        best, best_p = s, p
            params[idx] = best_p

        elif idx in TYPE2_CCI:
            best, best_p = -1, (PERIODS[0], CCI_LIMITS[0])
            for period in PERIODS:
                for lim in CCI_LIMITS:
                    p = (period, lim)
                    s = rule(p, OHLC)[0]
                    if s > best:
                        best, best_p = s, p
            params[idx] = best_p

        elif idx in TYPE3_RSI:
            n = len(RSI_LIMITS)
            best, best_p = -1, (PERIODS[0], RSI_LIMITS[1], RSI_LIMITS[0])
            for period in PERIODS:
                for lb in range(n - 1):
                    for ub in range(lb + 1, n):
                        p = (period, RSI_LIMITS[ub], RSI_LIMITS[lb])
                        s = rule(p, OHLC)[0]
                        if s > best:
                            best, best_p = s, p
            params[idx] = best_p

        elif idx in TYPE3_CCI:
            n = len(CCI_LIMITS)
            best, best_p = -1, (PERIODS[0], CCI_LIMITS[1], CCI_LIMITS[0])
            for period in PERIODS:
                for lb in range(n - 1):
                    for ub in range(lb + 1, n):
                        p = (period, CCI_LIMITS[ub], CCI_LIMITS[lb])
                        s = rule(p, OHLC)[0]
                        if s > best:
                            best, best_p = s, p
            params[idx] = best_p

        else:  # TYPE4: single period
            best, best_p = -1, PERIODS[0]
            for period in PERIODS:
                s = rule(period, OHLC)[0]
                if s > best:
                    best, best_p = s, period
            params[idx] = best_p

    return params


def get_features(df, selected_indices, rule_params):
    """Build feature DataFrame (logr + selected rule signals)."""
    OHLC = [df.Open, df.High, df.Low, df.Close]
    logr = np.log(df.Close / df.Close.shift(1))

    result = pd.DataFrame({'logr': logr})
    for idx in selected_indices:
        result[f'R{idx + 1}'] = ALL_RULES[idx](rule_params[idx], OHLC)[1]

    result.dropna(inplace=True)
    return result


def evaluate(features_df, weights):
    """Returns (total_return_pct, max_dd_pct, sharpe)."""
    w = np.array(weights).flatten()
    X = features_df.values[:, 1:]  # skip logr column

    position = (X @ w).flatten()
    max_abs = np.max(np.abs(position))
    if max_abs > 0:
        position /= max_abs

    logr = features_df['logr'].values
    port_r = logr * position

    total_return = float(port_r.sum() * 100)
    max_dd = float(min(port_r.cumsum()) * 100) if len(port_r) > 0 else 0.0
    sharpe = (float(port_r.mean() / port_r.std() * np.sqrt(48 * 365))
              if port_r.std() > 0 else 0.0)
    return total_return, max_dd, sharpe


def run_ga(features_df, generations, pop_size, n_parents):
    """Run GA and return best weights as 1D array."""
    result = ga_module.GA_train(
        features_df,
        optimizing_selection=2,
        sol_per_pop=pop_size,
        num_parents_mating=n_parents,
        num_generations=generations,
    )
    return np.array(result).flatten()


def run_mc(df, selected_indices, rule_params, weights, n_sims, noise_std):
    """Price-noise Monte Carlo. Returns % of sims with positive return."""
    positive = 0
    for seed in range(n_sims):
        np.random.seed(seed)
        noisy = df.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            noise = np.random.normal(0, noise_std, len(df))
            noisy[col] = df[col] * np.exp(noise)
        noisy['High'] = np.maximum(
            noisy['High'], np.maximum(noisy['Open'], noisy['Close'])
        )
        noisy['Low'] = np.minimum(
            noisy['Low'], np.minimum(noisy['Open'], noisy['Close'])
        )
        try:
            feat = get_features(noisy, selected_indices, rule_params)
            ret, _, _ = evaluate(feat, weights)
            if ret > 0:
                positive += 1
        except Exception:
            pass
    return positive / n_sims * 100


def engine_loop(config, state, stop_event):
    """
    Main background loop. Call in a daemon thread.
    config keys: data, selected_indices, train_pct, ga_generations,
                 ga_pop, ga_parents, num_mc_sims, mc_noise_std,
                 mc_pass_threshold, min_return, min_sharpe
    state: shared dict read by the UI
    stop_event: threading.Event
    """
    import matplotlib.pyplot as plt
    plt.ioff()

    data = config['data'].copy()
    selected = config['selected_indices']

    split = int(len(data) * config['train_pct'] / 100)
    train_data = data.iloc[:split].reset_index(drop=True)
    test_data = data.iloc[split:].reset_index(drop=True)

    state['total_bars'] = len(data)
    state['train_bars'] = len(train_data)
    state['test_bars'] = len(test_data)
    state['phase'] = 'training'
    state['winners'] = state.get('winners', [])

    # --- Phase 1: rule params (manual or trained) ---
    manual = config.get('manual_params')
    if manual:
        rule_params = manual
        state['status'] = 'Using manual rule parameters.'
    else:
        try:
            rule_params = train_selected_rules(train_data, selected, state)
        except Exception as e:
            state['status'] = f'Rule training failed: {e}'
            state['phase'] = 'stopped'
            return

    state['status'] = 'Building features...'
    try:
        train_features = get_features(train_data, selected, rule_params)
        test_features = get_features(test_data, selected, rule_params)
    except Exception as e:
        state['status'] = f'Feature generation failed: {e}'
        state['phase'] = 'stopped'
        return

    state['rule_params'] = {RULE_NAMES[k]: str(v) for k, v in rule_params.items()}
    state['phase'] = 'running'

    # Build the params list in order for saving (matching original winners.csv format)
    ordered_params = [rule_params[i] for i in selected]

    iteration = state.get('iteration', 0)

    # --- Phase 2: GA loop ---
    while not stop_event.is_set():
        iteration += 1
        state['iteration'] = iteration
        state['status'] = (
            f'Iteration {iteration} — GA running '
            f'({config["ga_generations"]} gen, pop {config["ga_pop"]})...'
        )

        try:
            weights = run_ga(
                train_features,
                config['ga_generations'],
                config['ga_pop'],
                config['ga_parents'],
            )

            ret, dd, sr = evaluate(test_features, weights)
            state['last_test'] = {
                'return': round(ret, 2),
                'dd': round(dd, 3),
                'sharpe': round(sr, 2),
            }
            state['status'] = (
                f'Iteration {iteration} — '
                f'Return: {ret:.1f}%  SR: {sr:.2f}  DD: {dd:.2f}%'
            )

            if ret >= config['min_return'] and sr >= config['min_sharpe']:
                state['status'] = (
                    f'Iteration {iteration} — Passed filter. '
                    f'Running MC ({config["num_mc_sims"]} sims)...'
                )

                mc_rate = run_mc(
                    test_data, selected, rule_params, weights,
                    config['num_mc_sims'], config['mc_noise_std'],
                )
                state['last_mc'] = round(mc_rate, 1)

                # Build a short verdict so you can see what failed
                verdict_parts = []
                if mc_rate < config['mc_pass_threshold']:
                    verdict_parts.append(f"MC {mc_rate:.0f}% < {config['mc_pass_threshold']:.0f}%")
                if ret < config['winner_min_return']:
                    verdict_parts.append(f"return {ret:.1f}% < {config['winner_min_return']:.1f}%")
                if sr < config['winner_min_sharpe']:
                    verdict_parts.append(f"Sharpe {sr:.2f} < {config['winner_min_sharpe']:.2f}")
                if abs(dd) > config['winner_max_dd']:
                    verdict_parts.append(f"DD {abs(dd):.1f}% > {config['winner_max_dd']:.1f}%")

                if verdict_parts:
                    state['status'] = f'Iteration {iteration} — Failed: {", ".join(verdict_parts)}'
                else:
                    state['status'] = f'Iteration {iteration} — All criteria met!'

                is_winner = (
                    mc_rate >= config['mc_pass_threshold']
                    and ret >= config['winner_min_return']
                    and sr >= config['winner_min_sharpe']
                    and abs(dd) <= config['winner_max_dd']
                )
                if is_winner:
                    w_hash = hashlib.md5(weights.tobytes()).hexdigest()[:12]
                    winner = {
                        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'iteration': iteration,
                        'rules_used': str([RULE_NAMES[i] for i in selected]),
                        'all_rules': str(ordered_params),
                        'all_weights': str(weights.tolist()),
                        'test_bars': len(test_data),
                        'return_pct': round(ret, 2),
                        'max_drawdown_pct': round(dd, 3),
                        'sharpe_ratio': round(sr, 3),
                        'mc_pass_rate': round(mc_rate, 1),
                        'strategy_hash': w_hash,
                    }
                    state['winners'].append(winner)
                    state['status'] = (
                        f'Iteration {iteration} — WINNER saved! '
                        f'Return: {ret:.1f}%  MC: {mc_rate:.1f}%'
                    )

                    # Append to CSV
                    winners_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), 'winners.csv'
                    )
                    pd.DataFrame([winner]).to_csv(
                        winners_path,
                        mode='a',
                        header=not os.path.exists(winners_path),
                        index=False,
                    )

        except Exception as e:
            state['status'] = f'Iteration {iteration} — Error: {e}'
            time.sleep(1)

    state['phase'] = 'stopped'
    state['status'] = 'Stopped.'
