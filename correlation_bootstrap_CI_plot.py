#!/usr/bin/env python3
"""
Correlation + bootstrap CI + scatter plot with regression line and 95% CI band.

Fixes ConstantInputWarning by skipping bootstrap resamples where x or y is constant.
Also produces a plot suitable for manuscript / supplement.

Outputs:
- n, Pearson r, R^2, p-value
- Bootstrap 95% CI for r and R^2 (percentile CI over valid resamples)
- Saves a PNG plot with regression line + 95% CI band

Usage:
  python correlation_bootstrap_CI_plot.py --csv GlyT1.csv --xcol exp --ycol pred --outfig GlyT1_corr.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def _safe_pearsonr(x, y):
    """Return r if defined, else np.nan."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return np.nan
    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return np.nan
    r, _ = pearsonr(x, y)
    return float(r)


def bootstrap_r_ci(x, y, n_boot=5000, seed=42, min_valid=1000, max_tries=200000):
    """
    Bootstrap 95% CI for Pearson r and R^2.
    Skips invalid bootstrap samples where correlation is undefined.

    Returns: (r_ci, r2_ci, n_valid, n_tries)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    r_vals = []
    tries = 0
    idx = np.arange(n)

    while len(r_vals) < n_boot and tries < max_tries:
        tries += 1
        samp = rng.choice(idx, size=n, replace=True)
        r = _safe_pearsonr(x[samp], y[samp])
        if np.isfinite(r):
            r_vals.append(r)

    r_vals = np.asarray(r_vals, dtype=float)
    if len(r_vals) < min_valid:
        # For very small n, you may not be able to get many valid resamples
        raise RuntimeError(
            f"Too few valid bootstrap resamples for CI: {len(r_vals)} valid out of {tries} tries. "
            f"Increase max_tries or reduce min_valid / n_boot."
        )

    r2_vals = r_vals ** 2
    r_ci = np.percentile(r_vals, [2.5, 97.5])
    r2_ci = np.percentile(r2_vals, [2.5, 97.5])
    return r_ci, r2_ci, int(len(r_vals)), int(tries)


def fit_line(x, y):
    """Simple linear regression y = a + b x using least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)  # [a, b]
    return float(beta[0]), float(beta[1])


def bootstrap_regression_band(x, y, x_grid, n_boot=5000, seed=42, min_valid=1000, max_tries=200000):
    """
    Bootstrap CI band for the regression mean prediction.
    Resamples (x,y) pairs, refits line, predicts on x_grid.
    Skips degenerate resamples where x is constant.

    Returns: (y_hat, y_lo, y_hi, n_valid, n_tries)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    idx = np.arange(n)

    # Point-estimate line
    a0, b0 = fit_line(x, y)
    y_hat = a0 + b0 * x_grid

    preds = []
    tries = 0
    while len(preds) < n_boot and tries < max_tries:
        tries += 1
        samp = rng.choice(idx, size=n, replace=True)
        xb = x[samp]
        yb = y[samp]
        if np.nanstd(xb) == 0.0:
            continue  # cannot fit slope
        a, b = fit_line(xb, yb)
        preds.append(a + b * x_grid)

    preds = np.asarray(preds, dtype=float)
    if len(preds) < min_valid:
        raise RuntimeError(
            f"Too few valid bootstrap resamples for regression band: {len(preds)} valid out of {tries} tries. "
            f"Increase max_tries or reduce min_valid / n_boot."
        )

    y_lo = np.percentile(preds, 2.5, axis=0)
    y_hi = np.percentile(preds, 97.5, axis=0)
    return y_hat, y_lo, y_hi, int(len(preds)), int(tries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV")
    ap.add_argument("--xcol", required=True, help="X column (e.g., experimental)")
    ap.add_argument("--ycol", required=True, help="Y column (e.g., predicted)")
    ap.add_argument("--outfig", default="correlation_plot.png", help="Output figure path (png)")
    ap.add_argument("--n_boot", type=int, default=5000, help="Bootstrap resamples (valid)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    x = pd.to_numeric(df[args.xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[args.ycol], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points.")

    # Point estimates
    r, p = pearsonr(x, y)
    r2 = r ** 2

    # Bootstrap CI for r and R^2 (skipping invalid resamples)
    r_ci, r2_ci, n_valid_r, tries_r = bootstrap_r_ci(
        x, y, n_boot=args.n_boot, seed=args.seed, min_valid=min(1000, args.n_boot)
    )

    # Regression line + CI band
    x_grid = np.linspace(np.min(x), np.max(x), 200)
    y_hat, y_lo, y_hi, n_valid_band, tries_band = bootstrap_regression_band(
        x, y, x_grid, n_boot=args.n_boot, seed=args.seed, min_valid=min(1000, args.n_boot)
    )

    # Print report
    print("Correlation analysis")
    print("--------------------")
    print(f"Sample size (n): {n}")
    print(f"Pearson r: {r:.3f}")
    print(f"R^2: {r2:.3f}")
    print(f"r 95% CI: [{r_ci[0]:.3f}, {r_ci[1]:.3f}]  (valid boot={n_valid_r}/{tries_r} tries)")
    print(f"R^2 95% CI: [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}]  (from r^2 boot)")
    print(f"Pearson r p-value: {p:.3e}")

    # Plot
    plt.figure(figsize=(5.2, 4.2))
    plt.scatter(x, y)
    plt.plot(x_grid, y_hat)
    plt.fill_between(x_grid, y_lo, y_hi, alpha=0.2)
    plt.xlabel(args.xcol)
    plt.ylabel(args.ycol)
    plt.title(f"n={n}, r={r:.3f} (95% CI {r_ci[0]:.3f}–{r_ci[1]:.3f}), R²={r2:.3f}")
    plt.tight_layout()
    plt.savefig(args.outfig, dpi=300)
    plt.close()

    print(f"Saved plot: {args.outfig}")


if __name__ == "__main__":
    main()
