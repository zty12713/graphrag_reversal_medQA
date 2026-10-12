"""
run_stats.py — Statistical significance testing for RAG ablation study.

Compares Baseline vs. "Revert all edges" and (optionally) Baseline vs.
"Ablation" using Wilcoxon signed-rank tests (paired, non-parametric) with
Benjamini-Hochberg FDR correction for multiple comparisons.

Requires that generation_eval.py was run with the updated code so that
output JSON files contain a "per_sample_scores" section.

Usage (LightRAG — 3 conditions, 18 tests):
    python run_stats.py \\
        --baseline  results/baseline_eval.json \\
        --revert    results/revert_edges_eval.json \\
        --ablation  results/ablation_eval.json \\
        --alpha     0.05

Usage (HippoRAG2 — 2 conditions, 9 tests; omit --ablation):
    python run_stats.py \\
        --baseline  results/baseline_eval.json \\
        --revert    results/revert_edges_eval.json \\
        --alpha     0.05

Output:
    - Prints a summary table to stdout
    - Saves stats_results.csv for use in your paper
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ── Helpers ────────────────────────────────────────────────────────────────

def load_per_sample(path: str) -> dict:
    """
    Load per_sample_scores from a generation_eval.py output file.

    Expected JSON structure (produced by updated generation_eval.py):
    {
      "Fact Retrieval": {
        "average_scores": {"rouge_score": 0.13, ...},
        "per_sample_scores": {
          "rouge_score":        {"q001": 0.08, "q002": 0.21, ...},
          "answer_correctness": {"q001": 0.30, "q002": 0.15, ...}
        }
      },
      ...
    }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for q_type, q_data in data.items():
        if "per_sample_scores" not in q_data:
            print(
                f"[ERROR] '{path}' has no 'per_sample_scores' for '{q_type}'.\n"
                f"        Re-run generation_eval.py with the updated code.",
                file=sys.stderr
            )
            sys.exit(1)
        out[q_type] = q_data["per_sample_scores"]
        # {metric: {sample_id: score}}
    return out


def aligned_pair(scores_a: dict, scores_b: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return two score arrays aligned by shared sample IDs.
    Only samples present in BOTH conditions are included (inner join).
    """
    shared = sorted(set(scores_a) & set(scores_b))
    if not shared:
        raise ValueError("No shared sample IDs between the two conditions.")
    a = np.array([scores_a[sid] for sid in shared], dtype=float)
    b = np.array([scores_b[sid] for sid in shared], dtype=float)
    return a, b


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test (two-sided).

    Returns:
        stat        — W statistic
        p_raw       — uncorrected p-value
        n           — number of paired samples used
        mean_diff   — mean(B) - mean(A)
        effect_r    — rank-biserial correlation r = Z / sqrt(N)
                      (|r| < 0.1 negligible, ~0.3 small, ~0.5 large)
    """
    diff = a - b
    n_nonzero = int(np.sum(diff != 0))

    if n_nonzero == 0:
        # Identical — no test possible
        return {
            "stat": np.nan, "p_raw": 1.0, "n": len(a),
            "mean_diff": 0.0, "effect_r": 0.0,
        }

    stat, pval = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")

    # Approximate Z for effect size (valid for n ≥ 10)
    z = abs(stats.norm.ppf(pval / 2))
    r = z / np.sqrt(len(a))

    return {
        "stat":      stat,
        "p_raw":     pval,
        "n":         len(a),
        "mean_diff": float(np.mean(b) - np.mean(a)),   # positive → B > A
        "effect_r":  round(r, 4),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank tests for RAG ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline",  required=True, help="Baseline eval JSON")
    parser.add_argument("--revert",    default=None,
                        help="Revert-all-edges eval JSON (optional; omit to run Ablation contrast only)")
    parser.add_argument("--ablation",  default=None,
                        help="Ablation eval JSON (optional; omit for frameworks with only Revert condition)")
    parser.add_argument("--alpha",     type=float, default=0.05,
                        help="Family-wise significance threshold after BH correction")
    parser.add_argument("--output",    default="stats_results.csv",
                        help="CSV path to save full results table")
    args = parser.parse_args()

    if args.revert is None and args.ablation is None:
        parser.error("At least one of --revert or --ablation must be provided.")

    baseline_scores = load_per_sample(args.baseline)

    contrasts = {}
    if args.revert is not None:
        contrasts["Baseline → Revert"]   = (baseline_scores, load_per_sample(args.revert))
    if args.ablation is not None:
        contrasts["Baseline → Ablation"] = (baseline_scores, load_per_sample(args.ablation))

    n_contrasts = len(contrasts)
    n_tests_expected = n_contrasts * 9   # 9 metric–question-type combinations
    print(f"\nRunning {n_contrasts} contrast(s) → up to {n_tests_expected} Wilcoxon tests "
          f"with BH correction (α = {args.alpha})")

    records = []
    for contrast_label, (data_a, data_b) in contrasts.items():
        for q_type in data_a:
            if q_type not in data_b:
                print(f"[WARN] Skipping '{q_type}': missing in second condition of '{contrast_label}'")
                continue
            for metric in data_a[q_type]:
                if metric not in data_b[q_type]:
                    print(f"[WARN] Skipping metric '{metric}' in '{q_type}': not in second condition")
                    continue
                try:
                    a, b = aligned_pair(data_a[q_type][metric], data_b[q_type][metric])
                except ValueError as e:
                    print(f"[WARN] {q_type} / {metric}: {e}")
                    continue

                res = wilcoxon_test(a, b)
                records.append({
                    "contrast":      contrast_label,
                    "question_type": q_type,
                    "metric":        metric,
                    "n":             res["n"],
                    "mean_A":        round(float(np.mean(a)), 4),
                    "mean_B":        round(float(np.mean(b)), 4),
                    "mean_diff":     round(res["mean_diff"], 4),
                    "p_raw":         round(res["p_raw"], 4),
                    "effect_r":      res["effect_r"],
                })

    if not records:
        print("[ERROR] No tests produced. Check that your JSON files share sample IDs.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # ── Benjamini-Hochberg FDR correction ──────────────────────────────────
    # Applied across ALL tests simultaneously (both contrasts × all metrics)
    reject, p_corr, _, _ = multipletests(df["p_raw"], method="fdr_bh", alpha=args.alpha)
    df["p_corrected"] = np.round(p_corr, 4)
    df["significant"] = reject

    # ── Print results ──────────────────────────────────────────────────────
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n" + "=" * 90)
    print("  WILCOXON SIGNED-RANK TEST RESULTS  (BH-corrected, α = {})".format(args.alpha))
    print("=" * 90)
    print("  mean_diff = mean(B) − mean(A)  |  positive → condition B scores higher")
    print("  effect_r  = rank-biserial r     |  <0.10 negligible, ~0.30 small, ~0.50 large")
    print()

    for contrast in df["contrast"].unique():
        sub = df[df["contrast"] == contrast].copy()
        print(f"── {contrast} ──")
        print(sub[["question_type", "metric", "n",
                    "mean_A", "mean_B", "mean_diff",
                    "p_raw", "p_corrected", "significant", "effect_r"
                    ]].to_string(index=False))
        print()

    # ── Summary ────────────────────────────────────────────────────────────
    sig = df[df["significant"]]
    print("=" * 90)
    if sig.empty:
        print(
            "✅  No statistically significant differences found after BH correction.\n"
            "    → Edge-reversed condition(s) perform equivalently to baseline on all metrics.\n"
            "    → This supports the interpretation that the framework is robust to edge directionality."
        )
    else:
        print(f"⚠️   {len(sig)} significant result(s) after BH correction (α = {args.alpha}):")
        print(sig[["contrast", "question_type", "metric",
                   "mean_diff", "p_corrected", "effect_r"]].to_string(index=False))

    # ── Save CSV ───────────────────────────────────────────────────────────
    df.to_csv(args.output, index=False)
    print(f"\n📄  Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
