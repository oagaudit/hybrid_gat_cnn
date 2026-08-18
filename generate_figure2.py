"""
generate_figure2_with_context.py
add Context points 
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.config_loader import CONFIG, get_project_root

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # <-- เพิ่มตรงนี้!

PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

cleaned_path = os.path.join(PROCESSED_DIR, 'brazil_cleaned.parquet')
df = pd.read_parquet(cleaned_path)
print(f"load data {len(df)} rows")

def get_pair_data_with_context(df, firm_a, firm_b):
    firm_a, firm_b = int(firm_a), int(firm_b)
    
    tenders_a = set(df[df['Competitors'] == firm_a]['Tender'])
    tenders_b = set(df[df['Competitors'] == firm_b]['Tender'])
    common_tenders = tenders_a.intersection(tenders_b)
    
    if not common_tenders:
        return [], [], []
    
    bids_a, bids_b = [], []
    context_x, context_y = [], []
    
    for tender in common_tenders:
        bid_a = df[(df['Tender'] == tender) & (df['Competitors'] == firm_a)]['Bid_norm'].values
        bid_b = df[(df['Tender'] == tender) & (df['Competitors'] == firm_b)]['Bid_norm'].values
        
        if len(bid_a) > 0 and len(bid_b) > 0:
            bids_a.append(bid_a[0])
            bids_b.append(bid_b[0])
            
            other_bidders = df[(df['Tender'] == tender) & 
                               (~df['Competitors'].isin([firm_a, firm_b]))]
            
            competitors_list = other_bidders['Competitors'].unique()
            for i in range(len(competitors_list)):
                for j in range(i+1, len(competitors_list)):
                    c1 = competitors_list[i]
                    c2 = competitors_list[j]
                    
                    bid_c1 = other_bidders[other_bidders['Competitors'] == c1]['Bid_norm'].values
                    bid_c2 = other_bidders[other_bidders['Competitors'] == c2]['Bid_norm'].values
                    
                    if len(bid_c1) > 0 and len(bid_c2) > 0:
                        context_x.append(bid_c1[0])
                        context_y.append(bid_c2[0])
    
    return bids_a, bids_b, context_x, context_y

bids_a_comp, bids_b_comp, ctx_x_comp, ctx_y_comp = get_pair_data_with_context(df, 50, 76)
bids_a_coll, bids_b_coll, ctx_x_coll, ctx_y_coll = get_pair_data_with_context(df, 69, 76)

print(f" 50_76: {len(bids_a_comp)} main, {len(ctx_x_comp)} context")
print(f" 69_76: {len(bids_a_coll)} main, {len(ctx_x_coll)} context")

# สร้าง Figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

ax1 = axes[0]
ax1.scatter(ctx_x_comp, ctx_y_comp, s=15, c='gray', alpha=0.3, edgecolors='none', label='Other pairs')
ax1.scatter(bids_a_comp, bids_b_comp, s=40, c='black', alpha=0.9, edgecolors='none', label='Pair (50,76)')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Normalized bid of firm A (50)', fontsize=11)
ax1.set_ylabel('Normalized bid of firm B (76)', fontsize=11)
ax1.set_title('Competitive pair (50_76)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2)
ax1.set_aspect('equal')
ax1.legend(fontsize=8)

ax2 = axes[1]
ax2.scatter(ctx_x_coll, ctx_y_coll, s=15, c='gray', alpha=0.3, edgecolors='none', label='Other pairs')
ax2.scatter(bids_a_coll, bids_b_coll, s=40, c='black', alpha=0.9, edgecolors='none', label='Pair (69,76)')

rect = Rectangle((0.0, 0.0), 0.2, 0.2, 
                 linewidth=2, edgecolor='red', facecolor='none', 
                 linestyle='--', label='Density gap')
ax2.add_patch(rect)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Normalized bid of firm A (69)', fontsize=11)
ax2.set_ylabel('Normalized bid of firm B (76)', fontsize=11)
ax2.set_title('Collusive pair (69_76)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.2)
ax2.set_aspect('equal')
ax2.legend(fontsize=8)

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'figure2_with_context.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n save: {output_path}")