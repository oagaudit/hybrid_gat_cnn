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
- [Paper](#paper)
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
- Nodes = tenders; an undirected edge connects two tenders sharing ≥ 1 bidder. The implementation scores candidate edges with Jaccard similarity over bidder sets (optionally with temporal decay from the `Date` column) and stores an `edge_weight`, but GATv2 is trained on **unweighted** edges — attention learns neighbour importance instead.
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

```
hybrid_gat_cnn/
├── config/
│   └── config.yaml                 # paths, countries, split ratios, image params
├── data/
│   ├── raw/                        # (not tracked) downloaded country CSVs
│   └── processed/                  # (not tracked) built by data_preprocessing.py
├── notebook/
│   └── explore_analysis.ipynb      # exploratory data analysis
├── paper/
│   ├── paper.tex                   # IEEE conference paper source
│   └── figs/                       # figures used by paper.tex
├── outputs/                        # (not tracked) all generated artefacts
│   ├── embeddings/                 # 64-dim pair embeddings: insample/ + fold_*/
│   ├── graph_data/                 # tender-level graphs (edges + screens)
│   ├── images/                     # 96x96 bid-rotation images
│   ├── logs/                       # per-step run logs
│   ├── models/                     # Stage 1 CNN: pooled_m1/ + cnn_cross_fold/fold_*/
│   ├── models_stage2/              # Stage 2 GATv2: in_sample/ + LOCO runs
│   ├── node_features/              # 135-dim features: insample/ + fold_*/
│   └── splits/                     # fixed train/val/test tender splits
├── src/
│   ├── models/
│   │   ├── bridge_module.py        # ContextualBridgeModule (Stage 1 -> 2 bridge)
│   │   ├── cnn_model.py            # BidRotationCNN / SimpleCNN (M1)
│   │   └── gatv2_model.py          # SimpleGAT (M2), GATv2Model (M3/M4)
│   ├── utils/
│   │   └── config_loader.py        # loads config.yaml, resolves project root
│   ├── create_tender_splits.py     # train/val/test tender splits
│   ├── data_preprocessing.py       # clean and normalise raw country data
│   ├── evaluate_baseline.py        # LR / RF classical baselines
│   ├── extract_embeddings.py       # frozen-CNN forward pass -> pair embeddings
│   ├── image_generator.py          # builds bid-rotation pair images
│   ├── prepare_graph_data.py       # builds tender graphs (edges, screens)
│   ├── prepare_node_features.py    # Bridge Module -> 135-dim node features
│   ├── train_cnn.py                # Stage 1 CNN training (pooled or LOCO fold)
│   ├── train_stage2.py             # GATv2 training (in-sample and LOCO)
│   └── ttest.py                    # paired t-tests
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

CUDA and CPU also work — pass `--device auto` or set the device in `config/config.yaml`.

Total runtime: ~2 h for Stage 1 (5 runs), ~30 min per Stage 2 configuration, ~10 h for the full set of experiments.

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

The pipeline runs in eight phases. Stage 1 is trained **four times** — once pooled (M1, in-sample) and once per LOCO fold — so that no target-country image ever reaches the CNN used to evaluate that country.

Create the output directories first:

```bash
mkdir -p outputs/logs
mkdir -p outputs/models/pooled_m1
mkdir -p outputs/models/cnn_cross_fold/fold_{brazil,japan,usa}
```

### Phase 0 — Data preprocessing

Loads the three country datasets, filters tenders by the minimum bidder count, and applies per-tender Min–Max normalisation.

```bash
python src/data_preprocessing.py
```

Output: `data/processed/{brazil,japan,usa}_cleaned.parquet`

### Phase 1 — Tender splits

Splits are created **before** any image is generated to prevent Stage 1 leakage.

```bash
python src/create_tender_splits.py
```

### Phase 2 — Bid-rotation images

Images are built from train tenders only.

```bash
python src/image_generator.py
```

### Phase 3 — Stage 1 CNN training

```bash
# Pooled M1 (in-sample): Brazil + Japan + USA
python src/train_cnn.py \
    --countries brazil japan usa \
    --images_dir ./outputs/images \
    --output_dir ./outputs/models/pooled_m1 \
    --model_type simple --use_class_weight \
    --n_runs 5 --epochs 50 \
    2>&1 | tee outputs/logs/step_m1_pooled.log
```

<details>
<summary><b>LOCO folds — three more runs</b></summary>

```bash
# Test = Brazil  -> train on Japan + USA
python src/train_cnn.py \
    --countries japan usa \
    --images_dir ./outputs/images \
    --output_dir ./outputs/models/cnn_cross_fold/fold_brazil \
    --model_type simple --use_class_weight \
    --n_runs 5 --epochs 50 \
    2>&1 | tee outputs/logs/step3_cnn_fold_brazil.log

# Test = Japan  -> train on Brazil + USA
python src/train_cnn.py \
    --countries brazil usa \
    --images_dir ./outputs/images \
    --output_dir ./outputs/models/cnn_cross_fold/fold_japan \
    --model_type simple --use_class_weight \
    --n_runs 5 --epochs 50 \
    2>&1 | tee outputs/logs/step3_cnn_fold_japan.log

# Test = USA  -> train on Brazil + Japan
python src/train_cnn.py \
    --countries brazil japan \
    --images_dir ./outputs/images \
    --output_dir ./outputs/models/cnn_cross_fold/fold_usa \
    --model_type simple --use_class_weight \
    --n_runs 5 --epochs 50 \
    2>&1 | tee outputs/logs/step3_cnn_fold_usa.log
```

</details>

### Phase 4 — Extract pair embeddings

The CNN is frozen; each fold uses its own checkpoint.

```bash
# M1 pooled (in-sample)
python src/extract_embeddings.py \
    --model_dir ./outputs/models/pooled_m1 \
    --model_file best_cnn.pth \
    --images_dir ./outputs/images \
    --countries_list brazil japan usa \
    --output_dir ./outputs/embeddings/insample \
    2>&1 | tee outputs/logs/step4_embeddings_insample.log
```

<details>
<summary><b>LOCO folds — three more runs</b></summary>

```bash
for FOLD in brazil japan usa; do
  python src/extract_embeddings.py \
      --model_dir ./outputs/models/cnn_cross_fold/fold_$FOLD \
      --model_file best_cnn.pth \
      --images_dir ./outputs/images \
      --countries_list brazil japan usa \
      --output_dir ./outputs/embeddings/fold_$FOLD \
      2>&1 | tee outputs/logs/step4_embeddings_fold_$FOLD.log
done
```

</details>

### Phase 5 — Graph data and node features

`prepare_graph_data.py` is independent of the CNN: it uses only the cleaned parquet files to build edges and the initial statistical-screen node features. Edges come from Jaccard similarity over bidder sets, with optional temporal decay from the `Date` column. Edge weights are computed but **not** consumed by the model — GATv2 runs on unweighted edges.

```bash
python src/prepare_graph_data.py
```

`prepare_node_features.py` then applies the Bridge Module to produce the 135-dim node features, once per embedding set:

```bash
# In-sample
python src/prepare_node_features.py \
    --embedding_csv ./outputs/embeddings/insample/all_pair_embeddings_with_labels.csv \
    --output_dir ./outputs/node_features/insample \
    2>&1 | tee outputs/logs/step_insample_node_features.log

# LOCO folds
for FOLD in brazil japan usa; do
  python src/prepare_node_features.py \
      --embedding_csv ./outputs/embeddings/fold_$FOLD/all_pair_embeddings_with_labels.csv \
      --output_dir ./outputs/node_features/fold_$FOLD \
      2>&1 | tee outputs/logs/step5_node_features_fold_$FOLD.log
done
```

### Phase 6 — Stage 2 training (in-sample ablation)

```bash
# M2 (SimpleGAT) / M3 (GATv2) / M4 (Hybrid)
for MODEL in simple_gat gatv2 hybrid; do
  python src/train_stage2.py \
      --model_type $MODEL \
      --node_features_dir ./outputs/node_features/insample \
      --output_dir ./outputs/models_stage2/in_sample \
      --device auto \
      --n_runs 5 --epochs 50 \
      2>&1 | tee outputs/logs/step7_in_sample_$MODEL.log
done
```

### Phase 6.2 — Classical baselines

```bash
python src/evaluate_baseline.py 2>&1 | tee outputs/logs/step_evaluate_baseline.log
```

### Phase 7 — Cross-market LOCO (3 folds × 3 conditions)

Conditions are set by flags: **C1** = no flag (zero-shot, local norm), **C2** = `--global_norm`, **C3** = `--global_norm --fine_tune_ratio 0.15`.

```bash
# Example: Test = Brazil, all three conditions
python src/train_stage2.py --model_type hybrid --test_country brazil \
    --node_features_dir ./outputs/node_features/fold_brazil \
    --output_dir ./outputs/models_stage2 \
    --n_runs 5 --epochs 100 \
    2>&1 | tee outputs/logs/step7_loco_brazil_c1.log

python src/train_stage2.py --model_type hybrid --test_country brazil \
    --node_features_dir ./outputs/node_features/fold_brazil \
    --output_dir ./outputs/models_stage2 \
    --global_norm \
    --n_runs 5 --epochs 100 \
    2>&1 | tee outputs/logs/step7_loco_brazil_c2.log

python src/train_stage2.py --model_type hybrid --test_country brazil \
    --node_features_dir ./outputs/node_features/fold_brazil \
    --output_dir ./outputs/models_stage2 \
    --global_norm --fine_tune_ratio 0.15 \
    --n_runs 5 --epochs 100 \
    2>&1 | tee outputs/logs/step7_loco_brazil_c3.log
```

<details>
<summary><b>Japan and USA folds</b></summary>

Same three commands with `--test_country japan` / `--node_features_dir ./outputs/node_features/fold_japan` (100 epochs), and `--test_country usa` / `--node_features_dir ./outputs/node_features/fold_usa` (50 epochs).

```bash
# Japan
for C in "c1:" "c2:--global_norm" "c3:--global_norm --fine_tune_ratio 0.15"; do
  TAG=${C%%:*}; FLAGS=${C#*:}
  python src/train_stage2.py --model_type hybrid --test_country japan \
      --node_features_dir ./outputs/node_features/fold_japan \
      --output_dir ./outputs/models_stage2 \
      $FLAGS --n_runs 5 --epochs 100 \
      2>&1 | tee outputs/logs/step7_loco_japan_$TAG.log
done

# USA
for C in "c1:" "c2:--global_norm" "c3:--global_norm --fine_tune_ratio 0.15"; do
  TAG=${C%%:*}; FLAGS=${C#*:}
  python src/train_stage2.py --model_type hybrid --test_country usa \
      --node_features_dir ./outputs/node_features/fold_usa \
      --output_dir ./outputs/models_stage2 \
      $FLAGS --n_runs 5 --epochs 50 \
      2>&1 | tee outputs/logs/step7_loco_usa_$TAG.log
done
```

</details>

### Phase 8 — Paired t-tests

```bash
python src/ttest.py 2>&1 | tee outputs/logs/step8_ttest.log
```

All metrics, checkpoints, and logs are written under `outputs/`.

---

## Configuration

Paths, country list, split ratios, and image-generation parameters live in `config/config.yaml` and are loaded through `src/utils/config_loader.py`.

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

## Paper

The condensed IEEE conference version of this work lives in `paper/paper.tex`, with its figures in `paper/figs/`. Build it with any standard LaTeX toolchain:

```bash
cd paper
latexmk -pdf paper.tex
```

Exploratory data analysis behind the paper is in `notebook/explore_analysis.ipynb`.

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
