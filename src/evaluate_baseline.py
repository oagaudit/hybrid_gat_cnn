"""
src/scripts/evaluate_baseline.py
Evaluate classical ML baselines (Logistic Regression, Random Forest) on the
SAME in-sample tender split used by Stage 1 (CNN) and Stage 2 (GATv2 M2/M3/M4),
to satisfy the "simple classical baseline on the same experimental setup"
requirement.

=====================================================================
FIXED VERSION:
- The previous version checked for train_mask/test_mask directly on the
  *_node_features.pt files. Those files are NEVER saved with masks
  (prepare_node_features.py only stores x, edge_index, y, tender_ids,
  country) — so the mask check always failed and the script silently fell
  back to an INDEPENDENT random train/test split, unrelated to
  create_tender_splits.py. This meant:
    1. The baseline was evaluated on a different set of tenders than
       M2/M3/M4, making the comparison unfair.
    2. Worse: a tender in this baseline's random "test" set could be one
       that Stage 1's CNN was trained on (tender-level leakage), exactly
       the problem the rest of this pipeline was fixed to avoid.
  This version loads the tender split directly from
  outputs/splits/{country}_split.json (same file Stage 1 and Stage 2 use)
  and maps it to node indices via tender_ids, exactly like
  train_stage2.py's in-sample branch does.

- Reports TWO feature variants per classifier, so the baseline can be
  compared fairly against BOTH ablation points:
    * "screens" (7-dim)   -> fair baseline for M2 (SimpleGAT) / M3 (GATv2),
      which also use only the 7 statistical screens.
    * "full" (135-dim)    -> fair baseline for M4 (Hybrid), which uses
      128-dim visual embeddings + 7 screens.
  Comparing a 135-dim classical baseline against a 7-dim GAT (as the
  previous version implicitly did) is not an apples-to-apples comparison.

- Reports mean ± std over 5 runs, matching the reporting convention used
  for M2/M3/M4/CNN. The tender split (train/val/test) is held FIXED across
  all 5 runs (identical to how Stage 1/Stage 2 fix the data split and vary
  only the model's random seed) — variance comes from the classifier's own
  stochastic elements (RF bootstrap sampling, LR solver's random init),
  not from re-splitting the data each run.
=====================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config_loader import CONFIG, get_project_root

PROJECT_ROOT = get_project_root()
NODE_FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs/node_features/insample")
SPLIT_DIR = os.path.join(PROJECT_ROOT, CONFIG['data'].get('splits_dir', 'outputs/splits'))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/models_stage2/in_sample")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRIES = ['brazil', 'japan', 'usa']
N_RUNS = 5
BASE_SEED = 42


def load_tender_split(country):
    """Load the SAME tender split Stage 1 (CNN) and Stage 2 (GATv2) use."""
    split_path = os.path.join(SPLIT_DIR, f"{country}_split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Tender split file not found for '{country}': {split_path}. "
            f"Run create_tender_splits.py first — the baseline MUST use the "
            f"same split as Stage 1/Stage 2, not an independent random split."
        )
    with open(split_path, 'r') as f:
        splits = json.load(f)
    return set(splits['train']), set(splits['val']), set(splits['test'])


def load_country_data_with_real_masks(country):
    """
    Load node features for one country and build train/val/test masks by
    matching each node's tender_id against outputs/splits/{country}_split.json
    — the exact same split used everywhere else in the pipeline.
    """
    file_path = os.path.join(NODE_FEATURES_DIR, f"{country}_node_features.pt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Node features not found: {file_path}")

    data = torch.load(file_path, map_location='cpu', weights_only=False)
    x = data.x.numpy() if hasattr(data, 'x') else data['x'].numpy()
    y = data.y.numpy() if hasattr(data, 'y') else data['y'].numpy()
    tender_ids = getattr(data, 'tender_ids', None) if hasattr(data, 'x') else data.get('tender_ids', None)

    if tender_ids is None or len(tender_ids) != len(y):
        raise RuntimeError(
            f"'{country}' node_features.pt is missing valid tender_ids "
            f"(got {0 if tender_ids is None else len(tender_ids)}, expected {len(y)}). "
            f"Cannot build a leakage-safe split without it. Re-run "
            f"prepare_graph_data.py / prepare_node_features.py."
        )

    train_set, val_set, test_set = load_tender_split(country)

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    unmatched = []
    for i, tid in enumerate(tender_ids):
        if tid in train_set:
            train_mask[i] = True
        elif tid in val_set:
            val_mask[i] = True
        elif tid in test_set:
            test_mask[i] = True
        else:
            unmatched.append(tid)

    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} tender(s) in '{country}' could not be matched to "
            f"train/val/test in {country}_split.json. First few: {unmatched[:5]}. "
            f"This usually means node_features.pt and the split file are out of "
            f"sync (regenerated at different times). Do not proceed with a "
            f"mismatched split."
        )

    return x, y, train_mask, val_mask, test_mask


def load_all_countries():
    """Load and concatenate node features + REAL leakage-safe masks for all countries."""
    all_x, all_y, all_train, all_val, all_test = [], [], [], [], []
    for country in COUNTRIES:
        x, y, train_mask, val_mask, test_mask = load_country_data_with_real_masks(country)
        all_x.append(x)
        all_y.append(y)
        all_train.append(train_mask)
        all_val.append(val_mask)
        all_test.append(test_mask)
        print(f"  {country}: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

    X = np.vstack(all_x)
    y = np.concatenate(all_y)
    train_mask = np.concatenate(all_train)
    val_mask = np.concatenate(all_val)
    test_mask = np.concatenate(all_test)
    return X, y, train_mask, val_mask, test_mask


def run_once(X_train, y_train, X_test, y_test, seed):
    """
    Train LR and RF with a given random seed (controls the classifier's own
    stochastic elements only — the train/test split itself does NOT change
    across runs, matching how Stage 1/Stage 2 fix the data split and vary
    only the model's weight-init seed).
    """
    lr = LogisticRegression(class_weight='balanced', max_iter=1000,
                             random_state=seed, solver='lbfgs')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_proba_lr = lr.predict_proba(X_test)[:, 1]

    rf = RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                 random_state=seed, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    def metrics(y_true, y_pred, y_proba):
        return {
            'test_acc': accuracy_score(y_true, y_pred),
            'test_prec': precision_score(y_true, y_pred, zero_division=0),
            'test_rec': recall_score(y_true, y_pred, zero_division=0),
            'test_f1': f1_score(y_true, y_pred, zero_division=0),
            'test_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
        }

    return metrics(y_test, y_pred_lr, y_proba_lr), metrics(y_test, y_pred_rf, y_proba_rf)


def evaluate_feature_variant(X, y, train_mask, test_mask, variant_name):
    """Run N_RUNS repetitions (fixed split, varying model seed) for one
    feature variant ('screens' or 'full') and return a results DataFrame."""
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n--- Feature variant: {variant_name} (dim={X.shape[1]}) ---")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Label distribution (train): 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")

    lr_rows, rf_rows = [], []
    for run_id in range(1, N_RUNS + 1):
        seed = BASE_SEED + run_id
        lr_metrics, rf_metrics = run_once(X_train, y_train, X_test, y_test, seed)
        lr_rows.append({'run_id': run_id, **lr_metrics})
        rf_rows.append({'run_id': run_id, **rf_metrics})
        print(f"  Run {run_id}: LR F1={lr_metrics['test_f1']:.4f} AUC={lr_metrics['test_auc']:.4f} | "
              f"RF F1={rf_metrics['test_f1']:.4f} AUC={rf_metrics['test_auc']:.4f}")

    lr_df = pd.DataFrame(lr_rows)
    rf_df = pd.DataFrame(rf_rows)
    return lr_df, rf_df


def summarize(df, model_name, variant_name):
    row = {'Model': model_name, 'Feature_set': variant_name}
    for metric in ['test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auc']:
        row[metric.replace('test_', '') + '_mean'] = df[metric].mean()
        row[metric.replace('test_', '') + '_std'] = df[metric].std()
    return row


def main():
    print("=" * 60)
    print("Baseline Evaluation (Leakage-safe: same tender split as Stage 1/2)")
    print("=" * 60)
    print("Loading node features and REAL train/val/test masks from "
          f"{SPLIT_DIR}/{{country}}_split.json ...")
    X_full, y, train_mask, val_mask, test_mask = load_all_countries()

    if X_full.shape[1] != 135:
        print(f"WARNING: expected 135-dim node features (128 visual + 7 screens), "
              f"got {X_full.shape[1]}. Check prepare_node_features.py output.")

    X_screens = X_full[:, -7:]   # fair baseline for M2 (SimpleGAT) / M3 (GATv2)
    X_hybrid = X_full            # fair baseline for M4 (Hybrid)

    summary_rows = []
    detailed_frames = {}

    for variant_name, X in [('screens', X_screens), ('full', X_hybrid)]:
        lr_df, rf_df = evaluate_feature_variant(X, y, train_mask, test_mask, variant_name)
        detailed_frames[f'LR_{variant_name}'] = lr_df
        detailed_frames[f'RF_{variant_name}'] = rf_df
        summary_rows.append(summarize(lr_df, 'LR (baseline)', variant_name))
        summary_rows.append(summarize(rf_df, 'RF (baseline)', variant_name))

        # Save detailed per-run CSVs, same convention as M2/M3/M4/CNN results
        lr_df.to_csv(os.path.join(OUTPUT_DIR, f'lr_baseline_{variant_name}_results.csv'), index=False)
        rf_df.to_csv(os.path.join(OUTPUT_DIR, f'rf_baseline_{variant_name}_results.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'baseline_results_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("Baseline Results Summary (mean ± std over 5 runs, fixed leakage-safe split)")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print("Saved per-run detail CSVs (lr_baseline_*.csv, rf_baseline_*.csv) to "
          f"{OUTPUT_DIR}")
    print("\nUse the 'screens' variant rows to compare against M2/M3, and the "
          "'full' variant rows to compare against M4 — comparing a 135-dim "
          "classical model against a 7-dim GAT (or vice versa) is not a fair "
          "comparison.")


if __name__ == "__main__":
    main()