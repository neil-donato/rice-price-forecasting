"""Forecasting Metro Manila retail rice prices (WFP / HDX Philippines Food Prices).

Pipeline
  1. Load + filter: retail price of "Rice (regular, milled)" in Metro Manila, one value per month.
  2. Models (all forecast 1..12 months ahead):
       - Naive, Seasonal naive           (benchmarks every other model must beat)
       - Holt-Winters (damped trend, additive seasonality) fitted on log price
       - Gradient boosting, direct multi-horizon, on lag / momentum / seasonality features
       - Equal-weight ensemble of Holt-Winters + gradient boosting
  3. Rolling-origin (expanding window) cross-validation on everything before the final 12 months.
  4. Model chosen by cross-validated MAE only; the last 12 months are a one-time holdout.
  5. 80% prediction intervals from cross-validated log errors, checked for coverage on the holdout.
  6. Refit on all data and forecast the next 12 months.

Run:  python src/forecast_rice.py   (from the project root)
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'wfp_food_prices_phl.csv'
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
COMMODITY, MARKET, PRICE_TYPE = 'Rice (regular, milled)', 'Metro Manila', 'Retail'
H = 12                 # forecast horizon (months)
CV_START = 168         # first CV origin has 14 years (168 months) of training data
CV_STEP = 3            # a new CV origin every 3 months
SEED = 42


# ----------------------------------------------------------------------------- data
def load_series():
    df = pd.read_csv(DATA, parse_dates=['date'])
    s = df[(df.commodity == COMMODITY) & (df.market == MARKET) & (df.pricetype == PRICE_TYPE)]
    assert s.duplicated(['date']).sum() == 0, 'expected one price per month'
    s = s.assign(month=s.date.dt.to_period('M').dt.to_timestamp()).set_index('month').price.sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq='MS')
    missing = full.difference(s.index)
    s = s.reindex(full).interpolate(method='time')   # linear fill of the few missing months
    s.index.freq = 'MS'
    return s, list(missing.strftime('%Y-%m'))


# ----------------------------------------------------------------------------- models
def naive(y_log, h=H):
    return np.repeat(y_log[-1], h)


def seasonal_naive(y_log, h=H):
    return np.array([y_log[len(y_log) - 12 + i] for i in range(h)])


def _hw_filter(params, y, m=12, return_state=False):
    """Damped-trend additive Holt-Winters recursion; returns SSE of one-step errors (or final state)."""
    a, b, g, phi = params
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    season = list(y[:m] - y[:m].mean())
    sse = 0.0
    for t in range(m, len(y)):
        pred = level + phi * trend + season[t - m]
        err = y[t] - pred
        sse += err * err
        new_level = a * (y[t] - season[t - m]) + (1 - a) * (level + phi * trend)
        trend = b * (new_level - level) + (1 - b) * phi * trend
        season.append(g * (y[t] - new_level) + (1 - g) * season[t - m])
        level = new_level
    if return_state:
        return level, trend, np.array(season[-m:])
    return sse


def holt_winters(y_log, h=H):
    """Damped-trend additive Holt-Winters on log prices; parameters by least squares (scipy)."""
    y = np.asarray(y_log, float)
    best = None
    for x0 in ([.5, .1, .1, .9], [.2, .05, .05, .95], [.8, .2, .2, .85]):
        r = minimize(_hw_filter, x0, args=(y,), method='L-BFGS-B',
                     bounds=[(.01, .99), (.001, .5), (.001, .99), (.8, .98)])
        if best is None or r.fun < best.fun:
            best = r
    phi = best.x[3]
    level, trend, season = _hw_filter(best.x, y, return_state=True)
    return np.array([level + sum(phi ** j for j in range(1, i + 1)) * trend + season[(i - 1) % 12]
                     for i in range(1, h + 1)])


def _features(y, t, h, start_month=1):
    """Features known at origin index t for a target h months ahead."""
    m_target = (t + h + start_month - 1) % 12
    m_origin = (t + start_month - 1) % 12
    w = y[t - 11:t + 1]
    return [y[t], y[t] - y[t - 1], y[t] - y[t - 3], y[t] - y[t - 6], y[t] - y[t - 12],
            y[t - 11:t + 1].mean() - y[t], y[t - 5:t + 1].mean() - y[t], w.std(),
            y[t - 1] - y[t - 2], h,
            np.sin(2 * np.pi * m_target / 12), np.cos(2 * np.pi * m_target / 12),
            np.sin(2 * np.pi * m_origin / 12), np.cos(2 * np.pi * m_origin / 12)]


def gradient_boosting(y_log, h=H):
    """One direct model for all horizons: predicts the log change y[t+h] - y[t]."""
    y = np.asarray(y_log, float)
    k = len(y)
    X, T = [], []
    for t in range(12, k - 1):
        for hh in range(1, min(h, k - 1 - t) + 1):
            X.append(_features(y, t, hh))
            T.append(y[t + hh] - y[t])
    model = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=.05, max_iter=250,
                                          max_depth=4, min_samples_leaf=25, l2_regularization=1.0,
                                          random_state=SEED)
    model.fit(np.array(X), np.array(T))
    Xp = np.array([_features(y, k - 1, hh) for hh in range(1, h + 1)])
    return y[-1] + model.predict(Xp)


MODELS = {'Naive': naive, 'Seasonal naive': seasonal_naive,
          'Holt-Winters (damped)': holt_winters, 'Gradient boosting': gradient_boosting}


def predict_all(y_log):
    preds = {name: fn(y_log) for name, fn in MODELS.items()}
    preds['Ensemble (HW + GB)'] = (preds['Holt-Winters (damped)'] + preds['Gradient boosting']) / 2
    return preds


# ----------------------------------------------------------------------------- evaluation
def evaluate(actual_price, pred_log):
    pred = np.exp(pred_log)
    err = actual_price - pred
    return np.abs(err), np.abs(err) / actual_price


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    s, missing = load_series()
    y_log = np.log(s.values)
    n = len(s)
    hold_k = n - H
    print(f'Series: {s.index[0]:%Y-%m} to {s.index[-1]:%Y-%m}, {n} months; '
          f'{len(missing)} missing months interpolated: {missing}')
    print(f'Holdout (never used for selection): {s.index[hold_k]:%Y-%m} to {s.index[-1]:%Y-%m}')

    # ---- rolling-origin cross-validation (everything strictly before the holdout)
    origins = list(range(CV_START, hold_k - H + 1, CV_STEP))
    print(f'Cross-validation: {len(origins)} origins, {s.index[origins[0]]:%Y-%m} to {s.index[origins[-1]]:%Y-%m}')
    rows, log_errs = [], {}
    for k in origins:
        preds = predict_all(y_log[:k])
        actual = s.values[k:k + H]
        for name, p in preds.items():
            ae, ape = evaluate(actual, p)
            log_errs.setdefault(name, []).append(y_log[k:k + H] - p)
            for h in range(H):
                rows.append((name, k, h + 1, ae[h], ape[h]))
    cv = pd.DataFrame(rows, columns=['model', 'origin', 'horizon', 'ae', 'ape'])
    cv_summary = cv.groupby('model').agg(MAE=('ae', 'mean'), MAPE=('ape', 'mean')).sort_values('MAE')
    cv_summary['MAPE'] *= 100
    sn_mae = cv_summary.loc['Seasonal naive', 'MAE']
    cv_summary['MAE vs seasonal naive'] = cv_summary.MAE / sn_mae - 1
    print('\nCROSS-VALIDATION (12-month-ahead, averaged over all horizons and origins)')
    print(cv_summary.round(3).to_string())
    best = cv_summary.index[0]
    print(f'\nSelected by CV MAE: {best}')
    cv_h = cv.groupby(['model', 'horizon']).ae.mean().unstack(0)

    # ---- one-time holdout evaluation
    preds_h = predict_all(y_log[:hold_k])
    actual_h = s.values[hold_k:]
    hold_rows = []
    for name, p in preds_h.items():
        ae, ape = evaluate(actual_h, p)
        hold_rows.append((name, ae.mean(), ape.mean() * 100))
    holdout = pd.DataFrame(hold_rows, columns=['model', 'MAE', 'MAPE']).set_index('model').sort_values('MAE')
    print('\nHOLDOUT (last 12 months, reported once, not used for selection)')
    print(holdout.round(3).to_string())

    # ---- 80% prediction intervals from CV log-errors of the selected model
    E = np.array(log_errs[best])                      # origins x horizons
    q10, q90 = np.quantile(E, .1, axis=0), np.quantile(E, .9, axis=0)
    ph = preds_h[best]
    lo, hi = np.exp(ph + q10), np.exp(ph + q90)
    coverage = float(np.mean((actual_h >= lo) & (actual_h <= hi)))
    print(f'\n80% interval coverage on holdout: {coverage:.0%} (target 80%)')

    # ---- final refit on all data, forecast next 12 months
    pf = predict_all(y_log)[best]
    future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq='MS')
    final = pd.DataFrame({'forecast': np.exp(pf), 'lower_80': np.exp(pf + q10),
                          'upper_80': np.exp(pf + q90)}, index=future_idx).round(2)
    final.index.name = 'month'
    print(f'\nFORECAST ({best}, PHP/kg)')
    print(final.to_string())

    # ---- save tables
    cv_summary.round(4).to_csv(OUT / 'cv_summary.csv')
    holdout.round(4).to_csv(OUT / 'holdout_summary.csv')
    cv_h.round(4).to_csv(OUT / 'cv_mae_by_horizon.csv')
    final.to_csv(OUT / 'forecast_next_12_months.csv')
    hold_table = pd.DataFrame({'actual': actual_h, 'forecast': np.exp(ph), 'lower_80': lo, 'upper_80': hi},
                              index=s.index[hold_k:]).round(2)
    hold_table.to_csv(OUT / 'holdout_forecast_vs_actual.csv')
    json.dump({'series': f'{COMMODITY}, {MARKET}, {PRICE_TYPE} (PHP/kg)',
               'months': int(n), 'missing_months_interpolated': missing,
               'cv_origins': len(origins), 'selected_model': best,
               'holdout_interval_coverage': coverage}, open(OUT / 'run_info.json', 'w'), indent=2)

    # ---- figures
    plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})
    blue, red, grey = '#1f5fa8', '#c0392b', '#8a8a8a'

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(s.index, s.values, color=blue, lw=1.6)
    for d, label, dy in [('2008-05', '2008 food price spike', 0), ('2018-09', 'Sep 2018 spike', 0),
                         ('2020-03', 'COVID-19 lockdowns begin', 2.2)]:
        ax.axvline(pd.Timestamp(d + '-01'), color=grey, ls=':', lw=1)
        ax.text(pd.Timestamp(d + '-01'), ax.get_ylim()[0] + .5 + dy, ' ' + label, fontsize=8, color=grey, va='bottom')
    ax.set_title(f'{COMMODITY}: retail price, {MARKET}, 2000 to {s.index[-1]:%Y}', loc='left')
    ax.set_ylabel('PHP per kg')
    fig.tight_layout(); fig.savefig(FIG / '01_price_history.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in cv_h.columns:
        ax.plot(cv_h.index, cv_h[name], marker='o', ms=3, lw=1.6 if name == best else 1,
                label=name, alpha=1 if name == best else .7)
    ax.set_xlabel('Months ahead'); ax.set_ylabel('MAE (PHP per kg)')
    ax.set_title('Cross-validated error by forecast horizon', loc='left'); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / '02_cv_error_by_horizon.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ctx = s.index >= s.index[hold_k - 24]
    ax.plot(s.index[ctx], s.values[ctx], color=blue, lw=1.8, label='Actual')
    ax.plot(s.index[hold_k:], np.exp(ph), color=red, lw=1.8, label=f'{best} forecast')
    ax.fill_between(s.index[hold_k:], lo, hi, color=red, alpha=.15, label='80% interval')
    ax.plot(s.index[hold_k:], np.exp(preds_h['Seasonal naive']), color=grey, ls='--', lw=1, label='Seasonal naive')
    ax.axvline(s.index[hold_k], color='k', lw=.8, ls=':')
    ax.set_title('Holdout: last 12 months, forecast made without seeing them', loc='left')
    ax.set_ylabel('PHP per kg'); ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / '03_holdout_forecast.png'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ctx = s.index >= s.index[-48]
    ax.plot(s.index[ctx], s.values[ctx], color=blue, lw=1.8, label='Actual')
    ax.plot(final.index, final.forecast, color=red, lw=1.8, label=f'{best} forecast')
    ax.fill_between(final.index, final.lower_80, final.upper_80, color=red, alpha=.15, label='80% interval')
    ax.set_title('Next 12 months (model refit on all data)', loc='left')
    ax.set_ylabel('PHP per kg'); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / '04_forecast_next_12_months.png'); plt.close(fig)


if __name__ == '__main__':
    main()
