import pandas as pd
import numpy as np

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
FILENAME_IN = "data/bias_evaluation_results.xlsx"
FILENAME_OUT = "data/bias_mitigation_metrics_ranked.xlsx"

# thresholds to evaluate "failure"
THRESHOLDS = [70, 50, 30]


def main():
    # --------------------------------------------------
    # Load raw results
    # --------------------------------------------------
    df = pd.read_excel(FILENAME_IN)

    # Identify attempt columns (attempt_0 ... attempt_24)
    attempt_cols = [c for c in df.columns if c.startswith("attempt_")]
    if "bias" not in df.columns:
        raise ValueError("Expected a 'bias' column in the file.")

    # --------------------------------------------------
    # Basic stats per bias across attempts
    # --------------------------------------------------
    # mean, std, min, max for each attempt column by bias
    stats = df.groupby("bias")[attempt_cols].agg(['mean', 'std', 'min', 'max'])
    stats.columns = ['_'.join(col) for col in stats.columns]  # flatten MultiIndex

    # --------------------------------------------------
    # Build per-bias overall metrics
    # --------------------------------------------------
    metrics = pd.DataFrame(index=stats.index)

    # Overall mean across attempts
    metrics["mean"] = stats[[c for c in stats.columns if c.endswith("_mean")]].mean(axis=1)
    # Overall std across attempts (average of per-attempt stds)
    metrics["std"] = stats[[c for c in stats.columns if c.endswith("_std")]].mean(axis=1)
    # Overall min/max across attempts
    metrics["min"] = stats[[c for c in stats.columns if c.endswith("_min")]].min(axis=1)
    metrics["max"] = stats[[c for c in stats.columns if c.endswith("_max")]].max(axis=1)


    # --------------------------------------------------
    # Global reference distribution (for relative bias)
    # --------------------------------------------------
    all_scores = df[attempt_cols].values.flatten()
    global_mean = np.mean(all_scores)
    global_std = np.std(all_scores) + 1e-8  # avoid division by zero

    metrics["relative_bias_abs"] = (metrics["mean"] - global_mean).abs()
    metrics["relative_bias_z"] = (metrics["mean"] - global_mean) / global_std

    # Sharpe-like ratio (if std is NaN, this will also be NaN)
    metrics["sharpe_ratio"] = metrics["mean"] / (metrics["std"] + 1e-8)

    # --------------------------------------------------
    # Failure rates per bias
    # --------------------------------------------------
    def failure_rate(sub_df, threshold):
        scores = sub_df[attempt_cols].values.flatten()
        return np.mean(scores < threshold)

    fail_70 = df.groupby("bias").apply(lambda x: failure_rate(x, 70))
    fail_50 = df.groupby("bias").apply(lambda x: failure_rate(x, 50))
    fail_30 = df.groupby("bias").apply(lambda x: failure_rate(x, 30))

    metrics["fail_rate_70"] = fail_70
    metrics["fail_rate_50"] = fail_50
    metrics["fail_rate_30"] = fail_30


    metrics["score"] = (
    metrics["mean"]
    - 30 * metrics["fail_rate_70"]
    - 20 * metrics["fail_rate_50"]
    - 10 * metrics["fail_rate_30"]
    )

    # --------------------------------------------------
    # RANKING
    # --------------------------------------------------
    # Multi-criteria sort:
    # 1) mean (descending)
    # 2) fail_rate_70 (ascending)
    # 3) relative_bias_z (descending)
    # 4) fail_rate_50 (ascending)
    # 5) fail_rate_30 (ascending)
    # 6) min (descending)
    ranked = metrics.sort_values(
        by=[
            "mean",
            "fail_rate_70",
            "relative_bias_z",
            "fail_rate_50",
            "fail_rate_30",
            "min",
        ],
        ascending=[False, True, False, True, True, False],
    )

    # Add explicit rank column (1 = best)
    ranked = ranked.reset_index().rename(columns={"index": "bias"})
    ranked.insert(0, "rank", ranked.index + 1)

    # --------------------------------------------------
    # Print & save
    # --------------------------------------------------
    print("\n=== Ranked Bias Mitigation Performance ===\n")
    print(ranked.round(4))

    # Save to Excel (ranked table)
    ranked.to_excel(FILENAME_OUT, index=False)


if __name__ == "__main__":
    main()
