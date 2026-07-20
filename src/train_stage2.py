"""
src/scripts/train_stage2.py
Stage 2: Train GAT-based models for tender-level collusion detection.

Supports:
- In-sample training (uses pre-defined tender splits from create_tender_splits.py)
- Cross-market (leave-one-country-out) with optional global normalization
- Few-shot fine-tuning (--fine_tune_ratio)
- Multiple runs for robustness
- Standard classification metrics (accuracy, precision, recall, f1, auc)
- Plots: Confusion Matrix, ROC Curve, Learning Curves
- Model specifications saved as JSON

=====================================================================
FIXED VERSION (2026-07-17):
- In-sample branch loads tender splits from create_tender_splits.py
  instead of using random train_test_split. This ensures that the same
  tender split used in Stage 1 (CNN) is also used in Stage 2 (GAT),
  preventing data leakage between train and test sets.
- tender_ids are preserved through merge_graphs() and load_graph_data()
  so that node-to-tender mapping is always available.
- Added logging (file + console) similar to train_cnn.py for consistency.

UPDATED (device / OOM handling):
- Added --device {auto,cpu,mps,cuda} to let the user pin the device
  explicitly instead of relying on the old "mps unless hybrid" heuristic.
- The GATv2 graph for the pooled in-sample setting can have a very large
  number of edges (Jaccard-similarity edges grow ~O(n^2) with bidder
  overlap), which can exceed the MPS backend's memory ceiling on Mac
  even when plenty of system RAM is free (MPS has its own allocator
  limit, separate from unified memory). When training on 'mps' hits a
  "MPS backend out of memory" RuntimeError, this script now automatically
  falls back to CPU for the rest of the run (and all subsequent runs)
  instead of crashing the whole job. CPU is slower but has no such
  hard ceiling, so it reliably finishes.
=====================================================================
"""

import os
import sys
import argparse
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gatv2_model import create_model
from src.utils.config_loader import CONFIG, get_project_root

PROJECT_ROOT = get_project_root()
SPLIT_DIR = os.path.join(PROJECT_ROOT, CONFIG['data'].get('splits_dir', 'outputs/splits'))


# ========================
# Utility functions
# ========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg):
    """
    Resolve the requested device.
      'cpu'  -> always CPU
      'cuda' -> CUDA if available, else error
      'mps'  -> MPS if available, else error
      'auto' -> cuda > mps > cpu
    """
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device('cuda')
    if device_arg == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available.")
        return torch.device('mps')
    # auto
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def is_oom_error(exc):
    """Detect out-of-memory errors from either MPS or CUDA backends."""
    msg = str(exc).lower()
    return ('out of memory' in msg) or ('mps backend out of memory' in msg) or ('cuda out of memory' in msg)


def load_graph_data(file_path, device='cpu'):
    """Load graph data (PyG Data object or dict) from .pt file."""
    data = torch.load(file_path, map_location=device, weights_only=False)
    if hasattr(data, 'x'):
        x = data.x
        edge_index = data.edge_index
        y = data.y
        edge_weight = getattr(data, 'edge_weight', None)
        tender_ids = getattr(data, 'tender_ids', None)
        country = getattr(data, 'country', 'unknown')
    else:
        x = data['x']
        edge_index = data['edge_index']
        y = data['y']
        edge_weight = data.get('edge_weight', None)
        tender_ids = data.get('tender_ids', None)
        country = data.get('country', 'unknown')
    return x, edge_index, y, edge_weight, tender_ids, country


def plot_learning_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.plot(history['val_f1'], label='Val F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.set_title('Validation Metrics')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ========================
# Load Tender Splits (for In-sample)
# ========================
def load_tender_splits(country_name):
    """Load tender splits JSON and return train/val/test tender sets."""
    split_path = os.path.join(SPLIT_DIR, f"{country_name}_split.json")
    if not os.path.exists(split_path):
        logging.warning(f"Split file not found: {split_path}")
        return None, None, None
    with open(split_path, 'r') as f:
        splits = json.load(f)
    return set(splits['train']), set(splits['val']), set(splits['test'])


# ========================
# Node-level training functions (with masks)
# ========================
def train_node(model, data, optimizer, criterion, device):
    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
    logits = model(x, edge_index, edge_weight)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask].to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred = logits.argmax(dim=1)
    acc = (pred[data.train_mask] == data.y[data.train_mask].to(device)).float().mean().item()
    return loss.item(), acc


def validate_node(model, data, mask, criterion, device):
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        edge_weight = data.edge_weight.to(device) if data.edge_weight is not None else None
        logits = model(x, edge_index, edge_weight)
        loss = criterion(logits[mask], data.y[mask].to(device))
        pred = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        y_prob = probs[mask].cpu().numpy()
        acc = (y_pred == y_true).mean()
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return loss.item(), acc, prec, rec, f1, auc_score, y_true, y_pred, y_prob


# ========================
# Merge multiple graphs into one (with metadata preservation)
# ========================
def merge_graphs(graphs):
    """
    Merge multiple PyG Data objects into one.
    Preserves tender_ids and country metadata for each node.
    """
    from torch_geometric.data import Data
    offset = 0
    all_x = []
    all_y = []
    all_edge_indices = []
    all_edge_weights = []
    all_tender_ids = []
    all_countries = []

    for g in graphs:
        all_x.append(g.x)
        all_y.append(g.y)

        # Get tender_ids (if available)
        tender_ids = getattr(g, 'tender_ids', None)
        if tender_ids is not None:
            all_tender_ids.extend(tender_ids)
        else:
            # If no tender_ids, use placeholder indices
            all_tender_ids.extend([f"{g.country}_{i}" for i in range(g.x.size(0))])

        all_countries.extend([g.country] * g.x.size(0))

        if g.edge_index is not None:
            edge_idx = g.edge_index + offset
            all_edge_indices.append(edge_idx)
            if hasattr(g, 'edge_weight') and g.edge_weight is not None:
                all_edge_weights.append(g.edge_weight)
        offset += g.x.size(0)

    x = torch.cat(all_x, dim=0)
    y = torch.cat(all_y, dim=0)
    edge_index = torch.cat(all_edge_indices, dim=1) if all_edge_indices else torch.empty(2, 0, dtype=torch.long)
    edge_weight = torch.cat(all_edge_weights, dim=0) if all_edge_weights else None

    merged = Data(x=x, edge_index=edge_index, y=y)
    if edge_weight is not None:
        merged.edge_weight = edge_weight

    # Preserve metadata
    merged.tender_ids = all_tender_ids
    merged.country_mapping = {i: all_countries[i] for i in range(len(all_countries))}
    merged.country_list = all_countries

    return merged


def log_graph_size_warning(merged_graph, out_dir):
    """Log node/edge counts and a heads-up if the graph is large enough
    that MPS OOM is likely. Purely informational — does not change behavior."""
    n_nodes = merged_graph.x.size(0)
    n_edges = merged_graph.edge_index.size(1)
    density = n_edges / (n_nodes ** 2) if n_nodes > 0 else 0.0
    logging.info(f"Graph size: {n_nodes} nodes, {n_edges} directed edge entries "
                 f"(density={density:.4f})")
    if n_edges > 200_000:
        logging.warning(f"Large graph ({n_edges} edges). If running on --device mps and you "
                         f"hit an 'MPS backend out of memory' error, this script will "
                         f"automatically retry on CPU. You can also pass --device cpu directly "
                         f"to skip the failed MPS attempt.")


# ========================
# Main
# ========================
def main():
    parser = argparse.ArgumentParser(description='Stage 2: GAT/GATv2 training')
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['simple_gat', 'gatv2', 'hybrid'],
                        help='M2: simple_gat, M3: gatv2, M4: hybrid')
    parser.add_argument('--node_features_dir', type=str, default='./outputs/node_features',
                        help='Directory with *_node_features.pt files')
    parser.add_argument('--output_dir', type=str, default='./outputs/models_stage2',
                        help='Base directory for saving models and results')
    parser.add_argument('--countries', type=str, nargs='+', default=['brazil', 'japan', 'usa'],
                        help='Countries to use (all)')
    parser.add_argument('--test_country', type=str, default=None,
                        help='If specified, leave-one-country-out; train on others, test on this country')
    parser.add_argument('--fine_tune_ratio', type=float, default=0.0,
                        help='Fraction of test country data to use for fine-tuning (few-shot). 0 = zero-shot.')
    parser.add_argument('--global_norm', action='store_true',
                        help='Apply global normalization (fit on training countries) to screen features')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation ratio from training countries (only used when test_country is set)')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Not used in cross-market')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--edge_dropout', type=float, default=0.2)
    parser.add_argument('--use_class_weight', action='store_true', default=True)
    parser.add_argument('--n_runs', type=int, default=5, help='Number of independent runs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'mps', 'cuda'],
                        help="Device to train on. 'auto' picks cuda > mps > cpu. If training on "
                             "'mps' or 'auto'-resolved-to-mps hits an out-of-memory error, this "
                             "script automatically falls back to CPU for the rest of the job "
                             "instead of crashing.")
    args = parser.parse_args()

    # Adjust output directory for cross-market
    if args.test_country:
        ft_tag = f"_ft{args.fine_tune_ratio}" if args.fine_tune_ratio > 0 else "_zeroshot"
        norm_tag = "_globalnorm" if args.global_norm else "_localnorm"
        out_dir = os.path.join(args.output_dir, 'cross_market',
                               f'{args.model_type}_test_{args.test_country}{ft_tag}{norm_tag}')
    else:
        out_dir = os.path.join(args.output_dir, 'in_sample')
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ============================================
    # ตั้งค่า logging (แสดงบนจอ + ไฟล์)
    # ============================================
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{args.model_type}_stage2.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("=" * 60)
    logging.info(f"Stage 2: {args.model_type.upper()} Training Started")
    logging.info("=" * 60)
    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Model type: {args.model_type}")
    logging.info(f"Node features dir: {args.node_features_dir}")
    if args.test_country:
        logging.info(f"LOCO mode: test_country = {args.test_country}")
        logging.info(f"Global norm: {args.global_norm}")
        logging.info(f"Fine-tune ratio: {args.fine_tune_ratio}")
    else:
        logging.info("In-sample mode")

    # Device selection
    try:
        device = resolve_device(args.device)
    except RuntimeError as e:
        logging.error(str(e))
        sys.exit(1)
    logging.info(f"Using device: {device} (requested: --device {args.device})")

    # Load all graph data
    from torch_geometric.data import Data
    all_graphs = []
    for country in args.countries:
        file_path = os.path.join(args.node_features_dir, f'{country}_node_features.pt')
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}. Skipping {country}.")
            continue
        x, edge_index, y, edge_weight, tender_ids, country_name = load_graph_data(file_path, device='cpu')
        data = Data(x=x, edge_index=edge_index, y=y)
        if edge_weight is not None:
            data.edge_weight = edge_weight
        data.country = country
        if tender_ids is not None:
            data.tender_ids = tender_ids
        all_graphs.append(data)

    if not all_graphs:
        logging.error("No graph data loaded. Exiting.")
        return

    # Merge all graphs into one (global graph)
    merged_graph = merge_graphs(all_graphs)
    num_nodes = merged_graph.x.size(0)

    # Determine input dimension
    if args.model_type == 'hybrid':
        in_dim = 135
        logging.info(f"Using hybrid features: {in_dim} dim")
    else:
        merged_graph.x = merged_graph.x[:, -7:]
        in_dim = 7
        logging.info(f"Using screens-only features: {in_dim} dim")
    logging.info(f"Node feature shape after adjustment: {merged_graph.x.shape}")

    log_graph_size_warning(merged_graph, out_dir)

    # Build masks for cross-market or in-sample
    if args.test_country:
        # ============================================
        # LOCO branch (unchanged — already correct)
        # ============================================
        logging.info("=" * 60)
        logging.info("LOCO mode: Building cross-market splits")
        logging.info("=" * 60)

        node_country = []
        offset = 0
        for g in all_graphs:
            n_nodes = g.x.size(0)
            node_country.extend([g.country] * n_nodes)
            offset += n_nodes
        node_country = np.array(node_country)

        target_indices = np.where(node_country == args.test_country)[0]
        source_indices = np.where(node_country != args.test_country)[0]

        from sklearn.model_selection import train_test_split

        if args.fine_tune_ratio > 0.0:
            target_labels = merged_graph.y[target_indices].numpy()
            target_ft_idx, target_test_idx = train_test_split(
                target_indices, train_size=args.fine_tune_ratio,
                stratify=target_labels, random_state=args.seed
            )
            train_mask_full_indices = np.concatenate([source_indices, target_ft_idx])
            test_mask_indices = target_test_idx
            logging.info(f"Few-Shot Transfer Learning: Using {args.fine_tune_ratio*100}% of {args.test_country} for training.")
        else:
            train_mask_full_indices = source_indices
            test_mask_indices = target_indices
            logging.info(f"Zero-Shot Cross-Market: No data from {args.test_country} used in training.")

        # Optional global normalization (fit on source indices only)
        if args.global_norm:
            logging.info("Applying global normalization on screen features (fit on training countries only).")
            if args.model_type == 'hybrid':
                visual = merged_graph.x[:, :-7]
                screens = merged_graph.x[:, -7:]
            else:
                visual = torch.empty(num_nodes, 0, dtype=merged_graph.x.dtype)
                screens = merged_graph.x
            source_screens = screens[source_indices]
            mean = source_screens.mean(dim=0, keepdim=True)
            std = source_screens.std(dim=0, keepdim=True) + 1e-8
            screens_norm = (screens - mean) / std
            if args.model_type == 'hybrid':
                merged_graph.x = torch.cat([visual, screens_norm], dim=1)
            else:
                merged_graph.x = screens_norm
            logging.info(f"Normalized screen features using mean {mean.squeeze().tolist()} and std {std.squeeze().tolist()}")

        train_mask_full = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask_full[train_mask_full_indices] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_mask_indices] = True

        train_indices = np.where(train_mask_full.numpy())[0]
        train_labels = merged_graph.y[train_indices].numpy()
        train_idx, val_idx = train_test_split(train_indices, test_size=args.val_ratio,
                                              stratify=train_labels, random_state=args.seed)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True

        merged_graph.train_mask = train_mask
        merged_graph.val_mask = val_mask
        merged_graph.test_mask = test_mask

        logging.info(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
        train_labels = merged_graph.y[train_mask].numpy()
        logging.info(f"Label distribution (train): 0={(train_labels==0).sum()}, 1={(train_labels==1).sum()}")

    else:
        # ============================================
        # IN-SAMPLE branch — Use pre-defined tender splits
        # ============================================
        logging.info("=" * 60)
        logging.info("In-sample mode: Loading tender splits from create_tender_splits.py")
        logging.info("=" * 60)

        # Build node-country mapping
        node_country = merged_graph.country_list
        if node_country is None:
            node_country = []
            for g in all_graphs:
                node_country.extend([g.country] * g.x.size(0))
            node_country = np.array(node_country)
        else:
            node_country = np.array(node_country)

        all_train_indices = []
        all_val_indices = []
        all_test_indices = []

        for g in all_graphs:
            country = g.country
            logging.info(f"\n📂 Processing {country.upper()}...")

            # Get tender IDs from graph
            tender_ids = getattr(g, 'tender_ids', None)
            if tender_ids is None:
                logging.warning(f"  No tender_ids found for {country}. Skipping.")
                continue

            # Load tender splits
            train_set, val_set, test_set = load_tender_splits(country)
            if train_set is None:
                logging.warning(f"  Split file not found for {country}. Using random split as fallback.")
                country_mask = np.array([c == country for c in node_country])
                country_indices = np.where(country_mask)[0]
                country_labels = merged_graph.y[country_indices].numpy()
                from sklearn.model_selection import train_test_split
                train_val_idx, test_idx = train_test_split(
                    country_indices, test_size=args.test_ratio,
                    stratify=country_labels, random_state=args.seed
                )
                train_val_labels = merged_graph.y[train_val_idx].numpy()
                val_ratio_adjusted = args.val_ratio / (1 - args.test_ratio)
                train_idx, val_idx = train_test_split(
                    train_val_idx, test_size=val_ratio_adjusted,
                    stratify=train_val_labels, random_state=args.seed
                )
                all_train_indices.extend(train_idx)
                all_val_indices.extend(val_idx)
                all_test_indices.extend(test_idx)
                continue

            # Build mapping: tender_id -> node index
            country_mask = np.array([c == country for c in node_country])
            country_indices = np.where(country_mask)[0]

            if len(country_indices) != len(tender_ids):
                logging.warning(f"  Mismatch: {len(country_indices)} nodes vs {len(tender_ids)} tender_ids for {country}")
                tender_to_idx = {}
                for idx, tid in enumerate(tender_ids):
                    if idx < len(country_indices):
                        tender_to_idx[tid] = country_indices[idx]
                    else:
                        for node_idx in country_indices:
                            if node_idx not in tender_to_idx.values():
                                tender_to_idx[tid] = node_idx
                                break
            else:
                tender_to_idx = {tid: country_indices[idx] for idx, tid in enumerate(tender_ids)}

            # Build train/val/test indices
            train_indices = [tender_to_idx[tid] for tid in train_set if tid in tender_to_idx]
            val_indices = [tender_to_idx[tid] for tid in val_set if tid in tender_to_idx]
            test_indices = [tender_to_idx[tid] for tid in test_set if tid in tender_to_idx]

            all_train_indices.extend(train_indices)
            all_val_indices.extend(val_indices)
            all_test_indices.extend(test_indices)

            logging.info(f"  ✅ {country}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

        # Create masks
        if all_train_indices and all_val_indices and all_test_indices:
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[all_train_indices] = True
            val_mask[all_val_indices] = True
            test_mask[all_test_indices] = True

            merged_graph.train_mask = train_mask
            merged_graph.val_mask = val_mask
            merged_graph.test_mask = test_mask

            logging.info(f"\n✅ In-sample splits loaded from tender splits:")
            logging.info(f"  Train: {train_mask.sum().item()}")
            logging.info(f"  Val: {val_mask.sum().item()}")
            logging.info(f"  Test: {test_mask.sum().item()}")
            logging.info(f"  Total: {num_nodes} "
                         f"({'All assigned ✅' if (train_mask | val_mask | test_mask).sum().item() == num_nodes else 'MISMATCH ⚠️'})")

            train_labels = merged_graph.y[train_mask].numpy()
            logging.info(f"  Label distribution (train): 0={(train_labels==0).sum()}, 1={(train_labels==1).sum()}")

        else:
            # Fallback: random split
            logging.warning("\n⚠️ Could not load tender splits. Using random split as fallback.")
            labels = merged_graph.y.numpy()
            indices = np.arange(num_nodes)
            from sklearn.model_selection import train_test_split
            train_val_idx, test_idx = train_test_split(indices, test_size=args.test_ratio,
                                                       stratify=labels, random_state=args.seed)
            train_val_labels = labels[train_val_idx]
            train_idx, val_idx = train_test_split(train_val_idx, test_size=args.val_ratio/(1-args.test_ratio),
                                                  stratify=train_val_labels, random_state=args.seed)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

            merged_graph.train_mask = train_mask
            merged_graph.val_mask = val_mask
            merged_graph.test_mask = test_mask

            logging.info(f"\n⚠️ Random split used (fallback):")
            logging.info(f"  Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")

    # ============================================
    # Training loop (multiple runs), with automatic MPS-OOM -> CPU fallback
    # ============================================
    all_results = []
    best_overall_f1 = -1
    best_model_state = None
    best_run_id = None
    best_history = None
    best_y_true = None
    best_y_pred = None
    best_y_prob = None

    run_id = 1
    fell_back_to_cpu = False
    while run_id <= args.n_runs:
        logging.info(f"\n{'='*60}\nRun {run_id}/{args.n_runs} (device={device})\n{'='*60}")
        set_seed(args.seed + run_id)

        model = create_model(model_type=args.model_type, in_dim=in_dim,
                             hidden_dim=args.hidden_dim, out_dim=2,
                             num_layers=args.num_layers, heads=args.heads,
                             dropout=args.dropout, edge_dropout=args.edge_dropout).to(device)

        # Class weights
        train_labels = merged_graph.y[merged_graph.train_mask].numpy()
        pos_count = (train_labels == 1).sum()
        neg_count = (train_labels == 0).sum()
        if args.use_class_weight and pos_count > 0:
            class_weights = torch.tensor([1.0, neg_count / pos_count], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logging.info(f"Using class weighting: class1 weight = {class_weights[1]:.2f}")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_val_f1 = 0.0
        best_epoch = -1
        best_model_state_local = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        try:
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train_node(model, merged_graph, optimizer, criterion, device)
                val_loss, val_acc, val_prec, val_rec, val_f1, val_auc, _, _, _ = validate_node(
                    model, merged_graph, merged_graph.val_mask, criterion, device)
                scheduler.step(val_loss)

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)

                logging.info(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                             f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    best_model_state_local = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break

            # Test evaluation
            model.load_state_dict(best_model_state_local)
            test_loss, test_acc, test_prec, test_rec, test_f1, test_auc, y_true, y_pred, y_prob = validate_node(
                model, merged_graph, merged_graph.test_mask, criterion, device)

        except RuntimeError as e:
            if is_oom_error(e) and device.type != 'cpu':
                logging.error(f"Run {run_id}: caught out-of-memory error on device '{device}': {e}")
                logging.error(f"Falling back to CPU for this run and all remaining runs "
                               f"(CPU has no hard memory ceiling like MPS/CUDA, just slower). "
                               f"You can avoid this delay next time by passing --device cpu directly.")
                device = torch.device('cpu')
                fell_back_to_cpu = True
                # Free whatever was allocated on the failed device before retrying
                del model, optimizer, scheduler
                if torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Retry the SAME run_id on CPU instead of advancing
                continue
            else:
                # Not an OOM error, or already on CPU (nothing left to fall back to) — re-raise
                raise

        results = {
            'run_id': run_id,
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_prec': test_prec,
            'test_rec': test_rec,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'device_used': str(device),
        }
        all_results.append(results)

        logging.info(f"\nRun {run_id} Test Results:")
        logging.info(f"  Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

        if test_f1 > best_overall_f1:
            best_overall_f1 = test_f1
            best_model_state = best_model_state_local
            best_run_id = run_id
            best_history = history
            best_y_true = y_true
            best_y_pred = y_pred
            best_y_prob = y_prob

        run_id += 1

    if fell_back_to_cpu:
        logging.warning("NOTE: one or more runs fell back to CPU after an out-of-memory error on "
                         "the original device. All reported runs below are still valid results — "
                         "the device only affects speed, not correctness — but if you need every "
                         "run trained on the exact same device for reproducibility notes, rerun "
                         "with --device cpu from the start.")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(out_dir, f'{args.model_type}_stage2_results.csv')
    results_df.to_csv(results_csv, index=False)
    logging.info(f"\nSaved results to {results_csv}")

    # Summary
    logging.info("\n" + "="*60)
    logging.info("SUMMARY OF ALL RUNS")
    logging.info("="*60)
    for metric in ['test_acc', 'test_prec', 'test_rec', 'test_f1', 'test_auc']:
        values = results_df[metric].values
        logging.info(f"{metric}: mean={values.mean():.4f} ± {values.std():.4f}")

    # Save best model
    best_model_path = os.path.join(out_dir, f'best_{args.model_type}_stage2.pth')
    torch.save({
        'model_state_dict': best_model_state,
        'args': vars(args),
        'best_run_id': best_run_id,
        'best_test_f1': best_overall_f1
    }, best_model_path)
    logging.info(f"\nBest model saved to {best_model_path}")

    # Generate plots
    if best_y_true is not None:
        cm = confusion_matrix(best_y_true, best_y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Competitive', 'Collusive'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - {args.model_type.upper()} (Best Run)')
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{args.model_type}.png'), dpi=150)
        plt.close()

        fpr, tpr, _ = roc_curve(best_y_true, best_y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {args.model_type.upper()}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'roc_curve_{args.model_type}.png'), dpi=150)
        plt.close()

    if best_history:
        plot_learning_curves(best_history, os.path.join(plots_dir, f'learning_curves_{args.model_type}.png'))

    # Model specifications
    temp_model = create_model(model_type=args.model_type, in_dim=in_dim,
                              hidden_dim=args.hidden_dim, out_dim=2,
                              num_layers=args.num_layers, heads=args.heads,
                              dropout=args.dropout, edge_dropout=args.edge_dropout)
    total_params = sum(p.numel() for p in temp_model.parameters())

    specs = {
        'model_type': args.model_type,
        'in_dim': in_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'heads': args.heads,
        'dropout': args.dropout,
        'edge_dropout': args.edge_dropout,
        'total_parameters': total_params,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss with class_weight',
        'use_class_weight': args.use_class_weight,
        'n_runs': args.n_runs,
        'global_norm': args.global_norm,
        'fine_tune_ratio': args.fine_tune_ratio,
        'requested_device': args.device,
        'fell_back_to_cpu': fell_back_to_cpu,
        'best_run_test_f1': best_overall_f1,
        'mean_test_f1': results_df['test_f1'].mean(),
        'std_test_f1': results_df['test_f1'].std(),
        'mean_test_auc': results_df['test_auc'].mean(),
        'std_test_auc': results_df['test_auc'].std(),
    }
    with open(os.path.join(plots_dir, f'model_specifications_{args.model_type}.json'), 'w') as f:
        json.dump(specs, f, indent=2)

    logging.info(f"\nPlots and specifications saved to {plots_dir}")
    logging.info("Stage 2 training completed.")


if __name__ == '__main__':
    main()