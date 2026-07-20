"""
Paired t-tests for the in-sample ablation study (Section 5.3).
...
"""

import os
import sys
import pandas as pd
from scipy import stats

# --- ใช้ get_project_root() ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import get_project_root

PROJECT_ROOT = get_project_root()
# -----------------------------

in_sample_dir = os.path.join(PROJECT_ROOT, 'outputs', 'models_stage2', 'in_sample')

paths = {
    'M2': os.path.join(in_sample_dir, 'simple_gat_stage2_results.csv'),
    'M3': os.path.join(in_sample_dir, 'gatv2_stage2_results.csv'),
    'M4': os.path.join(in_sample_dir, 'hybrid_stage2_results.csv'),
    'LR_screens': os.path.join(in_sample_dir, 'lr_baseline_screens_results.csv'),
    'RF_screens': os.path.join(in_sample_dir, 'rf_baseline_screens_results.csv'),
    'LR_full': os.path.join(in_sample_dir, 'lr_baseline_full_results.csv'),
    'RF_full': os.path.join(in_sample_dir, 'rf_baseline_full_results.csv'),
}

dfs = {}
missing = []
for name, p in paths.items():
    if not os.path.exists(p):
        missing.append((name, p))
    else:
        dfs[name] = pd.read_csv(p)

if missing:
    print("The following required result files were not found:")
    for name, p in missing:
        print(f"  {name}: {p}")
    print("\nRun train_stage2.py (for M2/M3/M4) and evaluate_baseline.py "
          "(for LR/RF) first.")
    exit(1)

out_dir = os.path.join(in_sample_dir, 'ttest')
os.makedirs(out_dir, exist_ok=True)


def compute_ttest(results_list, model_a_name, model_a_df, model_b_name, model_b_df, metric):
    t_stat, p_val = stats.ttest_rel(model_a_df[metric], model_b_df[metric])
    print(f"--- Compare {model_a_name} vs {model_b_name} ({metric}) ---")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}\n")
    results_list.append({
        'Comparison': f"{model_a_name} vs {model_b_name}",
        'Metric': metric,
        't_statistic': t_stat,
        'p_value': p_val
    })


# ============================================
# TABLE 1: CORE
# ============================================
print("=" * 60)
print("CORE comparisons (answer the architectural research questions)")
print("=" * 60)
core_results = []
for metric in ['test_f1', 'test_auc']:
    compute_ttest(core_results, "M4", dfs['M4'], "M2", dfs['M2'], metric)
    compute_ttest(core_results, "M4", dfs['M4'], "M3", dfs['M3'], metric)

core_df = pd.DataFrame(core_results)
core_csv = os.path.join(out_dir, 'paired_ttest_core.csv')
core_df.to_csv(core_csv, index=False)
print(f"Saved core results to {core_csv}\n")


# ============================================
# TABLE 2: SUPPLEMENTARY
# ============================================
print("=" * 60)
print("SUPPLEMENTARY comparisons (baseline discussion)")
print("=" * 60)
baseline_results = []
for metric in ['test_f1', 'test_auc']:
    compute_ttest(baseline_results, "M4", dfs['M4'], "RF (full)", dfs['RF_full'], metric)
    compute_ttest(baseline_results, "M4", dfs['M4'], "LR (full)", dfs['LR_full'], metric)
    compute_ttest(baseline_results, "M3", dfs['M3'], "RF (screens)", dfs['RF_screens'], metric)
    compute_ttest(baseline_results, "M3", dfs['M3'], "LR (screens)", dfs['LR_screens'], metric)
    compute_ttest(baseline_results, "M2", dfs['M2'], "RF (screens)", dfs['RF_screens'], metric)
    compute_ttest(baseline_results, "M2", dfs['M2'], "LR (screens)", dfs['LR_screens'], metric)

baseline_df = pd.DataFrame(baseline_results)
baseline_csv = os.path.join(out_dir, 'paired_ttest_baseline.csv')
baseline_df.to_csv(baseline_csv, index=False)
print(f"Saved supplementary baseline results to {baseline_csv}")

print("\nNOTE: comparisons against LR are mathematically a one-sample t-test "
      "against a constant (LR has zero variance across runs). This is expected.")