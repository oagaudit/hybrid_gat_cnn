import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path

# ==========================================
# 1. Base Path
# ==========================================
base_path = Path("outputs/models_stage2/cross_market")

print(f" Find CSV in: {base_path} ")

# find all .csv 
all_csv_files = list(base_path.rglob("*.csv"))

if not all_csv_files:
    print("No CSV")
else:
    print(f"All CSV {len(all_csv_files)} ")

results = []
processed_conditions = set() #  

# ==========================================
# 2. read file
# ==========================================
for filepath in all_csv_files:
    path_str = str(filepath).lower()
    
    if 'brazil' in path_str: country = 'Brazil'
    elif 'japan' in path_str: country = 'Japan'
    elif 'usa' in path_str: country = 'USA'
    else: continue  
        
    if 'localnorm' in path_str and 'zeroshot' in path_str:
        condition = 'C1: Zero-shot (Local Norm)'
    elif 'globalnorm' in path_str and 'zeroshot' in path_str:
        condition = 'C2: Zero-shot (Global Norm)'
    # FIXED: actual filenames use "ft0_15" (underscore), not "ft0.15" (dot).
    # The old check `'ft0.15' in path_str` never matched a real file, so every
    # few-shot (C3) result was silently skipped and Figure 5.13 was missing
    # the C3 bars for all three countries.
    elif 'globalnorm' in path_str and re.search(r'ft0[._]15', path_str):
        condition = 'C3: Few-shot 15% (Global Norm)'
    else:
        continue
        
    unique_key = f"{country}_{condition}"
    if unique_key in processed_conditions:
        print(f"  WARNING: skipping duplicate match for {unique_key} -> {filepath} "
              f"(already used a different file for this country/condition — check for "
              f"stale files left over from an earlier run)")
        continue
        
    try:
        df = pd.read_csv(filepath)
        if 'test_acc' not in df.columns:
            continue
            
        acc_mean, acc_std = df['test_acc'].mean(), df['test_acc'].std()
        f1_mean, f1_std = df['test_f1'].mean(), df['test_f1'].std()
        auc_mean, auc_std = df['test_auc'].mean(), df['test_auc'].std()
        
        results.append({
            'Target Country': country,
            'Condition': condition,
            'Accuracy': f"{acc_mean:.4f} ± {acc_std:.4f}",
            'F1-Score': f"{f1_mean:.4f} ± {f1_std:.4f}",
            'ROC-AUC': f"{auc_mean:.4f} ± {auc_std:.4f}",
            'F1_numeric': f1_mean,
            'F1_std': f1_std,
            'n_runs': len(df)
        })
        processed_conditions.add(unique_key)
        print(f"read file success {country} | {condition} (from {filepath.name})")
    except Exception as e:
        print(f"read file {filepath.name} no success: {e}")

df_results = pd.DataFrame(results)

# ==========================================
# 3. print
# ==========================================
if not df_results.empty:
    cond_order = ['C1: Zero-shot (Local Norm)', 'C2: Zero-shot (Global Norm)', 'C3: Few-shot 15% (Global Norm)']
    df_results['Condition'] = pd.Categorical(df_results['Condition'], categories=cond_order, ordered=True)
    df_results = df_results.sort_values(['Target Country', 'Condition'])
    
    print("\n=== Table Cross-Market (LOCO) model Hybrid (M4) ===")
    display_cols = ['Target Country', 'Condition', 'Accuracy', 'F1-Score', 'ROC-AUC']
    # NOTE: to_string() used instead of to_markdown() — the latter requires the
    # optional 'tabulate' package, which is not in requirements.txt (not part
    # of the environment used to produce the thesis results). Avoids adding a
    # new dependency just for console pretty-printing.
    print(df_results[display_cols].to_string(index=False))
    print("===========================================================================\n")
else:
    print("\n no file")

# ==========================================
# 4. create graph
# ==========================================
if not df_results.empty:
    sns.set_theme(style="whitegrid")

    countries = sorted(df_results['Target Country'].unique())
    conditions = cond_order  # fixed order: C1, C2, C3
    colors = ['#4C72B0', '#DD8452', '#55A868']

    n_countries = len(countries)
    n_conditions = len(conditions)
    bar_width = 0.8 / n_conditions
    x = np.arange(n_countries)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cond in enumerate(conditions):
        heights, errs = [], []
        for country in countries:
            row = df_results[(df_results['Target Country'] == country) &
                              (df_results['Condition'] == cond)]
            if row.empty:
                print(f"  WARNING: no data for {country} / {cond} — leaving gap in chart "
                      f"(check that all 9 LOCO runs — 3 countries x 3 conditions — "
                      f"were found under {base_path})")
                heights.append(np.nan)
                errs.append(0)
            else:
                heights.append(row['F1_numeric'].values[0])
                errs.append(row['F1_std'].values[0])

        offset = (i - (n_conditions - 1) / 2) * bar_width
        bars = ax.bar(x + offset, heights, bar_width, yerr=errs, capsize=3,
                       label=cond, color=colors[i % len(colors)],
                       edgecolor='black', alpha=0.9)

        for rect, h in zip(bars, heights):
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.4f}",
                            (rect.get_x() + rect.get_width() / 2, h),
                            ha='center', va='bottom',
                            xytext=(0, 9), textcoords='offset points', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.set_ylabel('Test F1-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Target Country (Leave-One-Country-Out)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Cross-market Setting', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

    plt.tight_layout()
    output_filename = base_path / 'figure_5_13_cross_market_f1.png'

    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"save file success: {output_filename}")

    n_expected = n_countries * n_conditions
    n_found = len(df_results)
    if n_found < n_expected:
        print(f"\n  WARNING: only found {n_found}/{n_expected} country-condition combinations. "
              f"The chart has gaps — check the printed table above for which are missing.")