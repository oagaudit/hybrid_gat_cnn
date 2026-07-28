# Hybrid GAT-CNN for Cross-Market Bid-Rigging Detection

A two-stage deep learning framework that combines **Convolutional Neural Networks (CNN)** and **Graph Attention Networks v2 (GATv2)** to detect bid-rigging (collusion) in public procurement, with a focus on **cross-market transferability**.

> **Paper:** *Hybrid GAT-CNN Architecture for Enhanced Cross-Market Bid-Rigging Detection*
> Mati Nakphon — University of Europe for Applied Sciences, Potsdam, Germany
> 📧 mati.nakphon@ue-germany.de, supernay26@gmail.com

---

## Table of Contents

- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Citation](#citation)
- [Ethical Use](#ethical-use)
- [License](#license)

---

## Motivation

Public procurement accounted for roughly **12.7% of global GDP in 2023**, and an estimated **10–25%** of government procurement budgets is lost to corruption. Bid rigging — bid rotation, bid suppression, and cover bidding — can inflate prices by **20–30%**.

Existing detection approaches have two recurring weaknesses:

1. **Single-modality learning.** CNN-based methods learn pairwise bidding geometry but ignore the multi-firm network structure; graph-based methods learn topology but discard the spatial patterns visible in bid-rotation images.
2. **Poor cross-market transfer.** Models trained in one procurement market degrade sharply in another because institutional rules differ (e.g. pre-announced reserve prices in Japan truncate bid distributions even under healthy competition).

This project addresses five gaps together: *(1)* no visual-spatial learning, *(2)* no graph topology modelling, *(3)* static attention, *(4)* weak cross-market robustness, *(5)* no few-shot / domain-stabilised learning.

---

## Key Contributions

- **Hybrid two-stage architecture** — CNN visual embeddings + GATv2 relational learning in one tender-level classifier.
- **Contextual Bridge Module (novel)** — a query-guided attention layer that resolves the granularity mismatch between *pair-level* CNN embeddings and *tender-level* graph nodes. The tender's seven statistical screens form the query; pair embeddings act as keys and values, so the model attends to the most anomalous bidder pairs instead of averaging them away.
- **GATv2 with GraphNorm, residual connections, and edge dropout** — dynamic attention that can re-rank neighbours per query node, unlike the static attention of GATv1.
- **Cross-market protocol** — Leave-One-Country-Out (LOCO) evaluation under three conditions: zero-shot with local normalisation (C1), zero-shot with global Z-score normalisation (C2), and few-shot fine-tuning with 15% target-market labels (C3).
- **Leakage-safe pipeline** — tender-level train/val/test splits are fixed *before* bid-rotation images are generated; the Stage 1 CNN never sees validation, test, or target-country images.

---

## Architecture

```
STAGE 1 — Offline Feature Extraction
┌──────────────┐   ┌───────────────────────┐   ┌──────────────────────┐
│ Raw bids     │ → │ Bid-rotation images   │ → │ CNN encoder (M1)     │
│ USA/JPN/BRA  │   │ 96 × 96 px per pair   │   │ → 64-dim embedding   │
└──────────────┘   └───────────────────────┘   └──────────────────────┘
        │                                                │
        │ 7 statistical screens                          │ embeddings cached
        │ (CV, SPD, DIFFP, RD, SKEW, KURT, KSTEST)       │ to disk, weights frozen
        ▼                                                ▼
STAGE 2 — Graph Relational Learning
┌────────────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Contextual Bridge Module   │ → │ GATv2 (2 layers) │ → │ MLP classifier   │
│ query = screens            │   │ 4 heads, GraphNorm│   │ 512 → 64 → 2     │
│ keys/values = pair embeds  │   │ residual, edge   │   │ collusive vs.    │
│ → 128-dim visual context   │   │ dropout p = 0.2  │   │ competitive      │
│ node feature = [128 ‖ 7]   │   │ → 512-dim node   │   │                  │
└────────────────────────────┘   └──────────────────┘   └──────────────────┘
```

**Stage 1 — Pairwise visual feature extraction**
- Images generated only for bidder pairs co-participating in **≥ 3 distinct tenders**.
- Each dot = one joint tender at `(x, y) = (Bid_norm,A, Bid_norm,B)`; the lower firm ID is always firm A for deterministic mapping. Contextual dots (other pairs in the same tender) are rendered at `α = 0.3`.
- Bids normalised per project with Min–Max scaling; 96 × 96 px @ 100 DPI, primary dot 10 px, contextual dot 4 px.
- CNN: 4 conv blocks (32 → 64 → 128 → 256, ReLU + 2×2 max-pool), FC-512 (dropout 0.5) → FC-256 → 64-dim linear embedding (no activation). **5,254,785 parameters.**
- Pair label = 1 only if **both** firms are verified cartel members.

**Stage 2 — Tender-level graph**
- Nodes = tenders; an undirected, unweighted edge connects two tenders sharing ≥ 1 bidder.
- Node label = 1 if at least one verified cartel member participated.
- Node feature `x_v = [f_v ‖ s_v] ∈ R^135` (128-dim visual context + 7 screens).
- Global Z-score normalisation for cross-market runs, with μ and σ computed on **training data only**.

### Model variants (ablation)

| Model | Features | Description |
|---|---|---|
| **M1** | bid-rotation images | CNN pair-level classifier / visual encoder |
| **M2** | 7 screens | SimpleGAT — GATv1 static attention, no GraphNorm / residual / edge dropout / bridge |
| **M3** | 7 screens | GATv2 + GraphNorm + residual + edge dropout |
| **M4** | 135 features | **Full hybrid** — M3 + Contextual Bridge Module |
| LR / RF | 7 or 135 | Classical baselines (scikit-learn defaults, balanced class weights) |

---

## Results

### Stage 1 — CNN (M1), pair-level test set (mean ± std, 5 runs)

| Metric | Best run | Mean ± std |
|---|---|---|
| Accuracy | 0.9048 | 0.8977 ± 0.0049 |
| Precision | 0.9296 | 0.9363 ± 0.0197 |
| Recall | 0.7968 | 0.7702 ± 0.0282 |
| F1-score | 0.8581 | 0.8445 ± 0.0109 |
| ROC-AUC | 0.9537 | 0.9533 ± 0.0025 |

### In-sample ablation (pooled test set, mean ± std over 5 runs)

| Model | Features | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| LR | 7 | 0.5631 | 0.1843 | 0.4938 | 0.2685 | 0.5361 |
| RF | 7 | 0.7683 | 0.3184 | 0.3728 | 0.3433 | 0.7034 |
| M2 (SimpleGAT) | 7 | 0.7355 | 0.1408 | 0.1704 | 0.1292 | 0.5712 |
| M3 (GATv2) | 7 | 0.5900 | 0.3281 | 0.8765 | 0.4584 | 0.7873 |
| LR | 135 | 0.8958 | 0.6593 | 0.7407 | 0.6977 | 0.9087 |
| RF | 135 | 0.9523 | **0.8932** | 0.8025 | **0.8453** | 0.9716 |
| **M4 (Hybrid)** | 135 | 0.9407 | 0.7582 | **0.9407** | 0.8386 | **0.9871** |

**Component contributions**
- M2 → M3 (dynamic attention + regularisation): **ΔF1 ≈ +0.33**, ΔAUC ≈ +0.22
- M3 → M4 (visual embeddings via Bridge Module): **ΔF1 ≈ +0.38**, ΔAUC ≈ +0.20 — the largest single gain
- Best M4 run: FPR 3.59%, FNR 6.17% (vs. M2: 16.75% / 60.49%)

**Significance (paired t-test, n = 5, α = 0.05):** M4 beats M2 and M3 on both F1 and ROC-AUC (all p < 0.05). Against RF (135 features), M4's ROC-AUC is significantly higher (p < 0.0001) and recall is higher, while the F1 difference is not significant (p = 0.7362).

### Cross-market generalisation (LOCO, model M4)

| Target | Condition | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Brazil | C1 zero-shot, local norm | 0.6020 | 0.1588 | 0.5356 |
| Brazil | C2 zero-shot, global norm | 0.6950 | 0.1219 | 0.7578 |
| Brazil | **C3 few-shot 15%** | 0.7512 | **0.6872** | 0.8599 |
| Japan | C1 | 0.4804 | 0.1354 | 0.3941 |
| Japan | C2 | 0.1456 | 0.2070 | 0.2906 |
| Japan | **C3** | 0.8586 | **0.5607** | 0.8853 |
| USA | C1 | 0.4929 | 0.3493 | 0.6711 |
| USA | C2 | 0.2787 | 0.3350 | 0.7153 |
| USA | **C3** | 0.9919 | **0.9783** | 0.9998 |

**Takeaway:** global normalisation alone does not remove domain shift and can even hurt (Japan's high recall / low precision under C2 means the model flagged almost everything). Fine-tuning on just **15%** of target-market labels recovers most of the gap in all three markets.

---

## Repository Structure

> Adjust the paths below to match your actual layout.

```
hybrid_gat_cnn/
├── data/
│   ├── raw/                  # source procurement CSVs (not tracked — see Data)
│   ├── processed/            # cleaned tenders, bids, screens
│   └── splits/               # fixed train/val/test indices (created BEFORE image generation)
├── images/                   # generated 96×96 bid-rotation images
├── embeddings/               # cached 64-dim pair embeddings (.h5)
├── src/
│   ├── preprocessing.py      # cleaning, min_bids ≥ 2 filter, normalisation
│   ├── make_splits.py        # stratified 70/15/15 tender-level splits
│   ├── bid_images.py         # bid-rotation image generation
│   ├── cnn.py                # Stage 1 CNN encoder (M1)
│   ├── bridge.py             # Contextual Bridge Module
│   ├── graph.py              # tender-level graph construction
│   ├── gatv2.py              # Stage 2 GATv2 + MLP head (M2/M3/M4)
│   ├── baselines.py          # Logistic Regression / Random Forest
│   ├── train_stage1.py
│   ├── train_stage2.py
│   ├── loco.py               # Leave-One-Country-Out experiments (C1/C2/C3)
│   └── evaluate.py           # metrics, paired t-tests, plots
├── configs/
├── results/                  # metrics, figures, t-SNE plots
├── notebooks/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/oagaudit/hybrid_gat_cnn.git
cd hybrid_gat_cnn

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Environment used in the paper**

| Component | Version |
|---|---|
| OS | macOS 26.5.2 (Apple M3, 16 GB unified memory) |
| Acceleration | Metal Performance Shaders (MPS) |
| Python | 3.13.3 |
| PyTorch | 2.11.0 |
| PyTorch Geometric | 2.7.0 |
| torchvision | 0.26.0 |
| NumPy / pandas | 2.4.4 / 3.0.2 |
| scikit-learn | 1.8.0 |
| matplotlib / seaborn | 3.10.8 / 0.13.2 |
| h5py / tqdm | 3.16.0 / 4.67.3 |

CUDA and CPU also work — set the device in `configs/`.

---

## Data

The datasets originate from the supplementary material of García Rodríguez et al. (2022) and include bid-rigging labels verified by regulatory authorities:

📦 https://doi.org/10.1016/j.autcon.2021.104047

Three markets were selected to represent contrasting conditions (Italy and Switzerland were excluded because of a missing date column and a missing bidder-ID column respectively):

| | Brazil | Japan | USA |
|---|---|---|---|
| Tenders after cleaning | 101 | 1,080 | 2,233 |
| Bids after cleaning | 683 | 13,515 | 5,483 |
| Avg. bidders per tender | 6.76 | 12.51 | 2.46 |
| % collusive tenders | 32.7% | 11.4% | 17.7% |
| Total bidder pairs | 122 | 8,676 | 223 |
| Collusive pairs | 60 | 3,165 | 80 |
| % collusive pairs | 49.2% | 36.5% | 35.9% |
| Graph nodes | 101 | 1,080 | 2,141 |
| Graph edges (undirected) | 1,553 | 44,511 | 581,116 |

*Notes:* only tenders with `min_bids ≥ 2` are kept; pair-level counts use `min_interactions ≥ 3`. The USA graph has 2,141 nodes because 92 isolated tenders (no shared bidders) were excluded from edges but retained in the feature matrix. Class imbalance is handled with class weights during training.

**Market characteristics** — USA (School Milk Program): sparse/bilateral competition, bid suppression and customer allocation. Japan (construction): multi-bidder tenders with strict bid-rotation patterns. Brazil (Petrobras, 2002–2013): high variance, incomplete cartels where collusive and non-collusive firms coexist.

Place the downloaded files in `data/raw/` before running the pipeline.

---

## Usage

```bash
# 1. Clean data and compute/attach the seven statistical screens
python src/preprocessing.py --config configs/default.yaml

# 2. Create fixed tender-level splits FIRST (prevents Stage 1 leakage)
python src/make_splits.py --ratio 70 15 15 --stratify

# 3. Generate bid-rotation images (training split only for CNN fitting)
python src/bid_images.py --min-interactions 3 --size 96 --dpi 100

# 4. Train the Stage 1 CNN encoder (M1) and cache pair embeddings
python src/train_stage1.py --seeds 43 44 45 46 47

# 5. Train Stage 2 models — the CNN stays frozen
python src/train_stage2.py --model M2   # SimpleGAT (GATv1)
python src/train_stage2.py --model M3   # GATv2 + GraphNorm/residual/edge dropout
python src/train_stage2.py --model M4   # Full hybrid with Bridge Module

# 6. Classical baselines
python src/baselines.py --features screens    # 7 features
python src/baselines.py --features full       # 135 features

# 7. Cross-market experiments
python src/loco.py --target japan --condition C1   # zero-shot, local norm
python src/loco.py --target japan --condition C2   # zero-shot, global norm
python src/loco.py --target japan --condition C3   # few-shot 15%, global norm

# 8. Aggregate metrics, paired t-tests, and figures
python src/evaluate.py --results results/
```

---

## Configuration

Stage 2 hyperparameters (identical for M2, M3, and M4 so differences come only from architecture):

| Parameter | Value |
|---|---|
| Learning rate | 1 × 10⁻³ |
| Weight decay | 10⁻⁵ |
| Dropout | 0.3 |
| Edge dropout | 0.2 |
| Hidden dimension | 128 |
| GAT layers | 2 |
| Attention heads per layer | 4 |
| Training mode | Full-batch (whole graph per forward pass) |
| Max epochs | 100 |
| Early stopping | Validation F1-score, patience 15 |
| Optimizer | Adam |
| LR scheduler | ReduceLROnPlateau on val loss (factor 0.5, patience 5) |
| Loss | CrossEntropyLoss with class weights |
| Independent runs | 5 (seeds 43–47) |

Stage 1: Adam (LR 1 × 10⁻³, weight decay 10⁻⁵), ReduceLROnPlateau (patience 5, factor 0.5), early stopping patience 15, batch size 32, max 100 epochs.

No grid or random search was performed — hyperparameters follow prior work and were checked on the validation set only.

---

## Reproducibility

- **Splits before images.** Tender-level train/val/test splits are created before any bid-rotation image is generated, so the Stage 1 CNN cannot leak information from validation or test tenders.
- **LOCO isolation.** A separate CNN is trained per target country using only training-split images from the two source countries. No target-country image is ever used for CNN training — including in the few-shot setting, where only the Stage 2 GATv2 classifier is fine-tuned.
- **Test set untouched.** The test set is never used for hyperparameter choice, model selection, or early stopping.
- **Five seeds (43–47)** for every stochastic model; Logistic Regression is deterministic and run once.
- **Significance testing** via paired t-tests (n = 5, α = 0.05) on F1 and ROC-AUC.

---

## Limitations

- Only three countries are covered, so results may not generalise to other auction formats (e.g. average-bid auctions).
- The graph depends on consistent, unobscured bidder identifiers — a fundamental constraint of real procurement data.
- In the Japan LOCO fold, the source countries supplied only **345** training pairs, far fewer than the other folds; this likely explains Japan's below-random zero-shot ROC-AUC.
- The seven statistical screens are used as provided in the source dataset without additional standardisation, which may disadvantage scale-sensitive classifiers such as Logistic Regression.
- Attention weights are not systematically extracted, so the model's internal reasoning remains largely a black box.

**Future work:** more countries and auction formats; a heterogeneous graph with bidders, tenders, agencies, regions, and contract types as typed nodes; attention-weight extraction for explainability; and testing whether fewer than 15% target-market labels suffice for adaptation.

---

## Citation

```bibtex
@article{nakphon2026hybrid,
  title   = {Hybrid GAT-CNN Architecture for Enhanced Cross-Market Bid-Rigging Detection},
  author  = {Nakphon, Mati},
  school  = {University of Europe for Applied Sciences},
  address = {Potsdam, Germany},
  year    = {2026}
}

@software{nakphon2026code,
  author  = {Nakphon, Mati},
  title   = {hybrid\_gat\_cnn: Source code for hybrid GAT-CNN bid-rigging detection},
  year    = {2026},
  url     = {https://github.com/oagaudit/hybrid_gat_cnn}
}
```

Please also cite the data source:

```bibtex
@article{garciarodriguez2022collusion,
  title   = {Collusion detection in public procurement auctions with machine learning algorithms},
  author  = {Garc{\'i}a Rodr{\'i}guez, Manuel J. and Rodr{\'i}guez-Montequ{\'i}n, Vicente and Ballesteros-P{\'e}rez, Pablo and Love, Peter E. D. and Signor, Regis},
  journal = {Automation in Construction},
  volume  = {133},
  pages   = {104047},
  year    = {2022},
  doi     = {10.1016/j.autcon.2021.104047}
}
```

Core method references: Huber & Imhof (2023) bid-rotation images; Imhof, Viklund & Huber (2025) graph attention with screens; Brody, Alon & Yahav (2022) GATv2; Cai et al. (2021) GraphNorm; Veličković et al. (2018) GAT; Rong et al. (2020) DropEdge.

---

## Ethical Use

This framework **detects potential bid-rigging activity from observable data only**. It does not establish legal evidence of collusion and does not determine criminal liability.

It is intended as a **screening and prioritisation tool** to support investigations by competition authorities and procurement agencies. Real-world deployment requires review by competition law specialists, and model predictions must not be used as direct legal evidence.

---

## License

Add a license file (e.g. MIT for code, CC BY 4.0 for documentation) and state it here. Note that the underlying datasets are governed by the terms of their original publication.
