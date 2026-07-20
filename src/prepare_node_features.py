"""
src/scripts/prepare_node_features.py
Prepare node features for GAT using Bridge Module:
- Load pair embeddings (CNN output) from specified CSV
- For each tender, aggregate its pair embeddings via ContextualBridgeModule
- Combine visual embedding (128-dim) with statistical screens (7-dim) -> node features (135-dim)
- Save graph data with updated node features for each country
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bridge_module import ContextualBridgeModule
from src.utils.config_loader import CONFIG, get_project_root

# ============================================
# ตั้งค่า PROJECT_ROOT และ PATH จาก config
# ============================================
PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
GRAPH_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['graph_dir'])
LOG_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['log_dir'])
MODEL_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['model_dir'])

# สร้างโฟลเดอร์ log (ถ้ายังไม่มี)
os.makedirs(LOG_DIR, exist_ok=True)


#logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'prepare_node_features.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
logging.info(f"Using device: {device}")


def load_pair_embeddings(csv_path):
    """Load pair embeddings from CSV file."""
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    pair_embeddings_map = {}
    for _, row in df.iterrows():
        pair_key = row['pair_key']
        emb = row[[f'emb_{i}' for i in range(64)]].values.astype(np.float32)
        pair_embeddings_map[pair_key] = emb
    return pair_embeddings_map


def process_country(country_name, pair_embeddings_map, bridge_module, output_dir):
    """Process a single country and save node features."""
    logging.info(f"\n📂 Processing {country_name.upper()}...")
    
    # Load cleaned data
    cleaned_path = os.path.join(PROCESSED_DIR, f"{country_name}_cleaned.parquet")
    if not os.path.exists(cleaned_path):
        logging.error(f"Cleaned file not found: {cleaned_path}")
        return
    
    df = pd.read_parquet(cleaned_path)
    
    # Get screen columns
    screen_cols = ['CV', 'SPD', 'DIFFP', 'RD', 'SKEW', 'KURT', 'KSTEST']
    missing = [c for c in screen_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing screens {missing} in {country_name}. Using available columns.")
        screen_cols = [c for c in screen_cols if c in df.columns]
    
    # Group by tender
    tender_groups = df.groupby('Tender')
    node_features_list = []
    labels_list = []
    tender_ids = []
    
    for idx, (tender_id, tender_df) in enumerate(tqdm(tender_groups, desc=f"Processing {country_name} tenders")):
        bidders = tender_df['Competitors'].values
        if len(bidders) < 2:
            continue
        
        # Generate pair keys
        pair_keys_in_tender = []
        for i in range(len(bidders)):
            for j in range(i+1, len(bidders)):
                id_a = int(float(bidders[i]))
                id_b = int(float(bidders[j]))
                if id_a <= id_b:
                    pair_key = f"{country_name}_{id_a}_{id_b}"
                else:
                    pair_key = f"{country_name}_{id_b}_{id_a}"
                pair_keys_in_tender.append(pair_key)
        
        # Collect embeddings
        pair_embeds = []
        for pk in pair_keys_in_tender:
            if pk in pair_embeddings_map:
                pair_embeds.append(pair_embeddings_map[pk])
            else:
                pair_embeds.append(np.zeros(64, dtype=np.float32))
        
        if not pair_embeds:
            continue
        
        pair_embeds_tensor = torch.tensor(np.array(pair_embeds), dtype=torch.float32).to(device)
        
        # Get screens
        screens = tender_df.iloc[0][screen_cols].values.astype(np.float32)
        screens_tensor = torch.tensor(screens, dtype=torch.float32).to(device).unsqueeze(0)
        
        # Compute visual embedding
        with torch.no_grad():
            visual_embed = bridge_module(pair_embeds_tensor.unsqueeze(0), screens_tensor)
            visual_embed = visual_embed.squeeze(0).cpu().numpy()
        
        # Concatenate visual (128) + screens (7) = 135
        node_feature = np.concatenate([visual_embed, screens])
        node_features_list.append(node_feature)
        
        label = int(tender_df['Collusive_competitor'].max())
        labels_list.append(label)
        tender_ids.append(tender_id)
    
    if len(node_features_list) == 0:
        logging.warning(f"No valid tenders for {country_name}")
        return
    
    # Load existing graph data to get edge_index
    graph_path = os.path.join(GRAPH_DIR, f"{country_name}_graph.pt")
    if not os.path.exists(graph_path):
        logging.error(f"Graph file not found: {graph_path}. Run prepare_graph_data.py first.")
        return
    
    graph_data = torch.load(graph_path, map_location='cpu', weights_only=False)
    if hasattr(graph_data, 'edge_index'):
        edge_index = graph_data.edge_index
        edge_weight = graph_data.edge_weight if hasattr(graph_data, 'edge_weight') else None
        original_tender_ids = graph_data.tender_ids if hasattr(graph_data, 'tender_ids') else None
    else:
        edge_index = graph_data['edge_index']
        edge_weight = graph_data.get('edge_weight', None)
        original_tender_ids = graph_data.get('tender_ids', None)
    
    # Align node ordering
    if original_tender_ids is not None:
        feature_map = {tid: node_features_list[i] for i, tid in enumerate(tender_ids)}
        ordered_features = []
        ordered_labels = []
        for tid in original_tender_ids:
            if tid in feature_map:
                ordered_features.append(feature_map[tid])
                ordered_labels.append(labels_list[tender_ids.index(tid)])
            else:
                ordered_features.append(np.zeros(135, dtype=np.float32))
                ordered_labels.append(0)
        node_features_np = np.array(ordered_features)
        labels_np = np.array(ordered_labels)
    else:
        node_features_np = np.array(node_features_list)
        labels_np = np.array(labels_list)
    
    # Convert to tensors
    x = torch.tensor(node_features_np, dtype=torch.float32)
    y = torch.tensor(labels_np, dtype=torch.long)
    
    # Save updated graph
    os.makedirs(output_dir, exist_ok=True)
    try:
        from torch_geometric.data import Data
        updated_graph = Data(x=x, edge_index=edge_index, y=y)
        if edge_weight is not None:
            updated_graph.edge_weight = edge_weight
        updated_graph.tender_ids = original_tender_ids if original_tender_ids is not None else tender_ids
        updated_graph.country = country_name
        out_path = os.path.join(output_dir, f"{country_name}_node_features.pt")
        torch.save(updated_graph, out_path)
        logging.info(f"  Saved updated graph to {out_path}")
    except ImportError:
        updated_graph = {
            'x': x,
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'y': y,
            'tender_ids': original_tender_ids if original_tender_ids is not None else tender_ids,
            'country': country_name
        }
        out_path = os.path.join(output_dir, f"{country_name}_node_features.pt")
        torch.save(updated_graph, out_path)
        logging.info(f"  Saved updated graph dict to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare node features with Bridge Module')
    parser.add_argument('--embedding_csv', type=str, required=True,
                        help='Path to pair embeddings CSV file (e.g., outputs/embeddings/all_pair_embeddings_with_labels.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: uses model_dir/node_features from config)')
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info(" Preparing Node Features with Bridge Module")
    logging.info("=" * 60)
    logging.info(f"Embedding CSV: {args.embedding_csv}")
    
    # กำหนด output_dir
    if args.output_dir is None:
        output_dir = os.path.join(MODEL_DIR, 'node_features')
    else:
        output_dir = args.output_dir
    logging.info(f"Output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    countries = ['brazil', 'japan', 'usa']
    
    # Load pair embeddings
    logging.info("\n Loading pair embeddings...")
    pair_embeddings_map = load_pair_embeddings(args.embedding_csv)
    if not pair_embeddings_map:
        logging.error("No embeddings loaded. Exiting.")
        return
    logging.info(f"Loaded {len(pair_embeddings_map)} pair embeddings")
    
    # Initialize Bridge Module
    bridge_module = ContextualBridgeModule(
        pair_embed_dim=64,
        screen_dim=7,
        visual_embed_dim=128,
        hidden_dim=64
    ).to(device)
    bridge_module.eval()
    logging.info("Bridge module initialized.\n")
    
    # Process each country
    for country in countries:
        process_country(country, pair_embeddings_map, bridge_module, output_dir)
    
    logging.info("\n Node features preparation completed.")


if __name__ == "__main__":
    main()