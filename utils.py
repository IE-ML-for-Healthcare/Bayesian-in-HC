"""Small helpers for 04_naive_bayes_opioid_demo_v2.ipynb

Threshold policy, same rule as the Responsible AI toolbox notebook (section 8):
1. Require recall of at least 60%
2. Require no more than 300 alerts per 1,000 patients
3. Among feasible thresholds, select the one with the highest precision
4. If none is feasible, say so, and apply a fallback that was declared in advance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def threshold_table(y_true, y_score):
    """Confusion counts, recall, precision and workload for every distinct score.

    A threshold only changes who gets flagged when it crosses a patient's score,
    so we test each distinct score instead of an arbitrary 0.01 grid.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rows = []
    for t in np.unique(np.r_[y_score, 1.0]):
        flag = y_score >= t
        tp = int((flag & (y_true == 1)).sum())
        fp = int((flag & (y_true == 0)).sum())
        fn = int(((~flag) & (y_true == 1)).sum())
        rows.append({
            "threshold": float(t),
            "TP": tp, "FP": fp, "FN": fn,
            "recall": tp / max(tp + fn, 1),
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "alerts_per_1000": 1000 * flag.mean(),
        })
    return pd.DataFrame(rows)


def select_threshold(y_true, y_score, min_recall=0.60, max_alerts_per_1000=300):
    """Apply the course threshold policy to validation scores.

    Primary rule: recall >= min_recall AND alerts <= capacity, then highest precision.
    Fallback (declared before looking at any test result): clinic capacity is a hard
    limit, so stay within capacity and catch as many cases as possible, then prefer
    higher precision.
    """
    tbl = threshold_table(y_true, y_score)
    within_capacity = tbl["alerts_per_1000"] <= max_alerts_per_1000
    feasible = tbl[within_capacity & (tbl["recall"] >= min_recall)]

    if not feasible.empty:
        best = feasible.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
        rule = "Primary rule met: recall floor and capacity both satisfied, highest precision"
    else:
        best = tbl[within_capacity].sort_values(
            ["recall", "precision", "threshold"], ascending=[False, False, False]).iloc[0]
        rule = (f"Primary rule NOT feasible: no threshold reaches {min_recall:.0%} recall "
                f"within {max_alerts_per_1000:.0f} alerts per 1,000. "
                "Fallback: capacity is a hard limit, so catch as many cases as capacity allows")

    return {"threshold": float(best["threshold"]), "rule": rule,
            "selected": best, "table": tbl, "feasible": not feasible.empty}


def plot_threshold_policy(choice, min_recall=0.60, max_alerts_per_1000=300, ax=None):
    """Recall vs workload, with the 'acceptable box' shaded: top-left of both limits."""
    tbl, sel = choice["table"], choice["selected"]
    ax = ax or plt.subplots(figsize=(7.5, 4.5))[1]
    ax.fill_between([0, max_alerts_per_1000], min_recall * 100, 100, color="#1b9e77", alpha=.12,
                    label="Acceptable box: enough cases caught, within capacity")
    ax.plot(tbl["alerts_per_1000"], tbl["recall"] * 100, c="k", lw=1.5, label="Each possible threshold")
    ax.axvline(max_alerts_per_1000, c="#d95f02", ls="--", lw=1, label=f"Capacity: {max_alerts_per_1000:.0f} reviews per 1,000")
    ax.axhline(min_recall * 100, c="#7570b3", ls="--", lw=1, label=f"Recall floor: {min_recall:.0%}")
    ax.scatter(sel["alerts_per_1000"], sel["recall"] * 100, s=90, c="#d62728", zorder=3,
               label=f"Selected threshold: {choice['threshold']:.3f}")
    ax.set(xlim=(0, 1000), ylim=(0, 102), xlabel="Reviews needed per 1,000 patients",
           ylabel="Recorded OUD cases caught (%)", title="Threshold policy on validation scores")
    ax.legend(fontsize=8, loc="lower right")
    return ax


def wilson_interval(successes, total, z=1.96):
    """95% interval for a proportion that behaves well with small counts."""
    if total == 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (centre - half, centre + half)
