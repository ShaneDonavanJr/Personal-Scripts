import numpy as np
from scipy.stats import skew, kurtosis, mode, kstest, norm

def analyze_array(
    x,
    percentiles=(1, 5, 25, 50, 75, 95, 99),
    alpha=0.05,
    decimals=4
):
    """
    Full descriptive statistics + Kolmogorov-Smirnov test for normality.
    Excel-style output.
    """

    # ---- Clean ----
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        print("No valid numeric data.")
        return

    n = x.size
    mean_val = x.mean()
    std_val = x.std(ddof=1)

    # ---- Descriptive stats ----
    stats = {
        "Count": n,
        "Sum": x.sum(),
        "Mean": mean_val,
        "Standard Error": std_val / np.sqrt(n),
        "Median": np.median(x),
        "Mode": mode(x, keepdims=False).mode,
        "Standard Deviation": std_val,
        "Sample Variance": np.var(x, ddof=1),
        "Skewness": skew(x),
        "Kurtosis": kurtosis(x),  # excess kurtosis
        "Minimum": x.min(),
        "Maximum": x.max(),
        "Range": x.max() - x.min(),
        **{f"Percentile {p}%": np.percentile(x, p) for p in percentiles},
    }

    # ---- Kolmogorov-Smirnov Test (Normality) ----
    z = (x - mean_val) / std_val  # standardize
    ks_stat, p_value = kstest(z, "norm")

    critical_value = 1.36 / np.sqrt(n)  # alpha = 0.05
    outcome = (
        "Reject – Not Normal"
        if ks_stat > critical_value
        else "Fail to Reject – Treat as Normal"
    )

    # ---- Pretty printing ----
    width = max(len(k) for k in stats.keys())
    fmt = f"{{:>{width}}} : {{:.{decimals}f}}"

    print("\nDescriptive Statistics")
    print("-" * (width + 15))
    for k, v in stats.items():
        if isinstance(v, (int, np.integer)):
            print(f"{k:>{width}} : {v}")
        else:
            print(fmt.format(k, v))

    # ---- KS Test Output ----
    # print("\nKolmogorov-Smirnov Results")
    # print("-" * (width + 15))
    # print(f"{'Testing significance level':>{width}} : {alpha}")
    # print(fmt.format("D-Statistic (K-S)", ks_stat))
    # print(fmt.format("Critical Value", critical_value))
    # print(fmt.format("P-Value", p_value))
    # print(f"{'Outcome':>{width}} : {outcome}")


import numpy as np

def goal_maker(score, x, targets=(80, 90, 95, 99), decimals=2, tie_method="mean"):
    """
    Compare a single score to a population array and print:
      - percentile rank of the score
      - the score thresholds for target percentiles (e.g., 80/90/95/99)

    tie_method:
      - "weak"  : percentile = % of values <= score
      - "strict": percentile = % of values <  score
      - "mean"  : average of strict and weak (nice for ties; default)
    """
    # ---- clean ----
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if x.size == 0:
        print("⚠️ No valid numeric data in population.")
        return

    s = float(score)
    n = x.size

    # ---- percentile rank of the score (empirical CDF style) ----
    lt = np.sum(x < s)
    le = np.sum(x <= s)

    if tie_method == "strict":
        pct_rank = 100.0 * (lt / n)
    elif tie_method == "weak":
        pct_rank = 100.0 * (le / n)
    elif tie_method == "mean":
        pct_rank = 100.0 * (((lt + le) / 2) / n)
    else:
        raise ValueError('tie_method must be one of: "strict", "weak", "mean"')

    # ---- target percentile thresholds ----
    targets = tuple(sorted(set(targets)))
    thresholds = {p: float(np.percentile(x, p)) for p in targets}

    # ---- pretty print ----
    print("\nGoal Maker")
    print("-" * 40)
    print(f"Population size (n)      : {n}")
    print(f"Your score               : {s:.{decimals}f}")
    print(f"Percentile of your score : {pct_rank:.{decimals}f}th percentile (tie_method='{tie_method}')")

    print("\nTarget percentiles (score needed)")
    print("-" * 40)
    for p in targets:
        print(f"{p:>3}% percentile score     : {thresholds[p]:.{decimals}f}")

    # ---- quick coaching line ----
    # Find next target above current percentile
    higher = [p for p in targets if pct_rank < p]
    if higher:
        next_p = min(higher)
        gap = thresholds[next_p] - s
        print("\nNext milestone")
        print("-" * 40)
        if gap <= 0:
            print(f"You're already at/above the {next_p}% mark.")
        else:
            print(f"To reach {next_p}%, aim for ~{thresholds[next_p]:.{decimals}f} (need +{gap:.{decimals}f}).")
    else:
        print("\nNext milestone")
        print("-" * 40)
        print(f"You’re already at/above the highest target ({max(targets)}%).")

    # Return useful stuff too (in case you want to log it)
    # return {
    #     "n": n,
    #     "score": s,
    #     "percentile_rank": pct_rank,
    #     "targets": thresholds,
    #     "tie_method": tie_method,
    # }