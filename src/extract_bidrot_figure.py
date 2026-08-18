"""
generate_figure2.py
สร้าง Figure 2 สำหรับวิทยานิพนธ์:
- ซ้าย: Competitive pair (50_76)
- ขวา: Collusive pair (69_76)
- แกน X = Normalized bid of firm A
- แกน Y = Normalized bid of firm B
ใช้ข้อมูลจาก cleaned parquet โดยตรง
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.config_loader import CONFIG, get_project_root

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

PROJECT_ROOT = get_project_root()
PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed_dir'])
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# 1. โหลดข้อมูลบราซิลที่ cleaned แล้ว
# ============================================
cleaned_path = os.path.join(PROCESSED_DIR, 'brazil_cleaned.parquet')
if not os.path.exists(cleaned_path):
    print(f"❌ ไม่พบไฟล์: {cleaned_path}")
    print("กรุณารัน preprocessing ก่อน")
    exit()

df = pd.read_parquet(cleaned_path)
print(f"✅ โหลดข้อมูล {len(df)} rows")

# ============================================
# 2. ฟังก์ชันดึงข้อมูล Bid_norm ของคู่บริษัท
# ============================================
def get_pair_data(df, firm_a, firm_b):
    """
    ดึงข้อมูล Bid_norm ของคู่ firm_a และ firm_b 
    จากทุก Tender ที่ทั้งคู่เข้าร่วม
    """
    # แปลงเป็น int เพื่อความแน่นอน
    firm_a = int(firm_a)
    firm_b = int(firm_b)
    
    # หา Tenders ที่ทั้งคู่เข้าร่วม
    tenders_a = set(df[df['Competitors'] == firm_a]['Tender'])
    tenders_b = set(df[df['Competitors'] == firm_b]['Tender'])
    common_tenders = tenders_a.intersection(tenders_b)
    
    if not common_tenders:
        print(f"⚠️ ไม่พบ Tender ที่ {firm_a} และ {firm_b} เข้าร่วมพร้อมกัน")
        return [], []
    
    # ดึง Bid_norm ของทั้งคู่ในแต่ละ Tender
    bids_a = []
    bids_b = []
    
    for tender in common_tenders:
        bid_a = df[(df['Tender'] == tender) & (df['Competitors'] == firm_a)]['Bid_norm'].values
        bid_b = df[(df['Tender'] == tender) & (df['Competitors'] == firm_b)]['Bid_norm'].values
        
        if len(bid_a) > 0 and len(bid_b) > 0:
            bids_a.append(bid_a[0])
            bids_b.append(bid_b[0])
    
    return bids_a, bids_b

# ============================================
# 3. ดึงข้อมูลของคู่ 69_76 และ 50_76
# ============================================
# คู่ Competitive (50_76)
bids_a_comp, bids_b_comp = get_pair_data(df, 50, 76)
print(f"✅ คู่ 50_76: {len(bids_a_comp)} interactions")

# คู่ Collusive (69_76)
bids_a_coll, bids_b_coll = get_pair_data(df, 69, 76)
print(f"✅ คู่ 69_76: {len(bids_a_coll)} interactions")

if len(bids_a_comp) == 0 or len(bids_a_coll) == 0:
    print("❌ ไม่มีข้อมูลเพียงพอสำหรับคู่ที่ต้องการ")
    print("   ลองตรวจสอบว่า MIN_INTERACTIONS ใน config.yaml ต่ำพอหรือไม่")
    exit()

# ============================================
# 4. สร้าง Figure
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# ---- ซ้าย: Competitive Pair (50_76) ----
ax1 = axes[0]
ax1.scatter(bids_a_comp, bids_b_comp, s=30, c='black', alpha=0.8, edgecolors='none')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel('Normalized bid of firm A (50)', fontsize=11)
ax1.set_ylabel('Normalized bid of firm B (76)', fontsize=11)
ax1.set_title('Competitive pair (50_76)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# ---- ขวา: Collusive Pair (69_76) ----
ax2 = axes[1]
ax2.scatter(bids_a_coll, bids_b_coll, s=30, c='black', alpha=0.8, edgecolors='none')
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Normalized bid of firm A (69)', fontsize=11)
ax2.set_ylabel('Normalized bid of firm B (76)', fontsize=11)
ax2.set_title('Collusive pair (69_76)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# ---- ปรับแต่งโดยรวม ----
plt.tight_layout()

# บันทึกภาพ
output_path = os.path.join(OUTPUT_DIR, 'figure2_bidrotation.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ บันทึก Figure 2 ที่: {output_path}")
print(f"   - คู่ 50_76: {len(bids_a_comp)} จุด (Competitive)")
print(f"   - คู่ 69_76: {len(bids_a_coll)} จุด (Collusive)")
print("\n📊 รูปนี้มีแกน X/Y และเหมาะสำหรับใส่ในวิทยานิพนธ์แล้ว!")