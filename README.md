# Multimodal Sepsis Prediction Framework

This repository provides a comprehensive, mathematically rigorous pipeline for evaluating sepsis outcomes by integrating clinical time-series data with peripheral blood transcriptomic profiles. The framework evaluates isolated unimodal baselines and probes the limits of cross-modal representation alignment using advanced manifold learning and optimal transport.

---

## Project Overview

Sepsis is characterized by complex interactions between host immune responses and physiological decline. This project evaluates predictive models using two distinct, highly heterogeneous data modalities:

* **Clinical Data (Physiology):** Electronic Health Record (EHR) data focusing on cardiovascular, respiratory, and routine lab indicators.
* **Genomic Data (Transcriptomics):** High-dimensional microarray and RNA-seq profiles focusing on early-stage innate immune markers.

Rather than relying purely on standard late-fusion, this framework utilizes **Topological Data Analysis (TDA)** and **Optimal Transport (OT)** to investigate whether the latent geometries of these two modalities can be mathematically harmonized to form a unified predictive space.

---

## Core Methodology

The analytical pipeline is divided into three distinct phases:

### 1. Unimodal Baselines
* **The ICU Doctor (Clinical):** An XGBoost-based physiological baseline trained on the PhysioNet 2019 Challenge dataset. It utilizes sparsity-aware gradient boosting to handle variable-length ICU stays and missing bedside data.
* **The Molecular Geneticist (Genomic):** A high-dimensional (7,902 genes) XGBoost baseline trained on curated Gene Expression Omnibus (GEO) datasets, designed to capture complex, non-linear transcriptomic interactions.

### 2. Topological Harmonization
To evaluate the viability of Early Fusion, the pipeline maps both modalities into deep latent spaces and attempts to align them:
* **Batch Correction:** Utilizes Domain-Adversarial Neural Networks (DANN) and **Harmony** to collapse cohort-specific technical acquisition biases (domain shift).
* **Trajectory Mapping:** Utilizes **PHATE** to map the continuous physiological decay of clinical patients, preserving global temporal structure better than standard UMAP/t-SNE.
* **Cross-Modal Alignment:** Employs Entropic Fused **Gromov-Wasserstein Optimal Transport (GW-OT)** to attempt a purely geometric, label-blind alignment between the clinical and genomic manifolds.

### 3. Rigorous Ablation & Validation Suite
The repository includes a comprehensive set of validation notebooks to stress-test the models:
* **Feature Parsimony:** Testing model fragility under extreme feature compression.
* **Probabilistic Calibration:** Evaluating overconfidence via Brier scores.
* **Temporal Degradation:** Simulating real-time bedside prospective inference to prove early-warning capabilities.
* **Algorithmic Fairness:** Stratifying performance across age and sex demographics.
* **LOCO Validation:** Leave-One-Cohort-Out testing to quantify susceptibility to cross-center domain shift.

---

## Repository Structure

```plaintext
multimodal_sepsis/
│
├── data/                            # Local data mounts (gitignored)
│   ├── raw/                         # Raw PhysioNet .psv and GEO Soft files
│   └── processed/                   
│       ├── clinical_tensors/        # Parsed ICU tabular matrices
│       ├── ml_tensors/              # Standardized transcriptomic matrices
│       └── mapped_matrices/         # Individual cohort matrices for LOCO
│
├── src/                             # Core parsers and inference logic
│   ├── clinical_datasets/           # EHR parsing scripts
│   ├── models/                      # DNN architectures and benchmarking
│   └── 00_multimodal_inference_engine.py 
│
├── manifolds/                       # Topological Alignment Pipeline
│   └── 01_genomic_clinical_manifold_dann.ipynb
│
├── notebooks/                       # Clinical Validation Suite
│   ├── 02_feature_parsimony.ipynb
│   ├── 03_reliability_and_calibration.ipynb
│   ├── 04_temporal_degradation.ipynb
│   ├── 05_algorithmic_fairness.ipynb
│   ├── 06_loco_external_validation.ipynb
│   └── 07_full_model_loco_validation.ipynb
│
├── outputs/                         # Generated artifacts
│   ├── models/                      # Trained XGBoost (.json) & PyTorch (.pth) weights
│   └── figures/                     # UMAPs, PHATE trajectories, ROC curves
│
├── docker-compose.yml               # Container orchestration
└── Dockerfile                       # Environment definition

Technical Stack
Machine Learning: XGBoost, PyTorch, Scikit-learn

Manifold Learning & Topology: UMAP, PHATE (phate), Harmony (harmonypy)

Optimal Transport: Python Optimal Transport (POT / ot)

Data Processing & Viz: Pandas, NumPy, Matplotlib, Seaborn

Infrastructure: NVIDIA CUDA-enabled Docker containerization.

Reproducibility Guide
To fully replicate the environment and execute the experimental pipeline from scratch, follow these steps:

Step 1: Environment Initialization
The entire pipeline is containerized to prevent dependency conflicts. Ensure Docker and NVIDIA Container Toolkit are installed.

# Build and launch the container
docker-compose up -d --build

# Access the interactive workspace
docker exec -it sepsis_multimodal_container /bin/bash

Step 2: Data Preparation
(Note: You must download the raw PhysioNet 2019 dataset and GEO cohorts into data/raw/ prior to this step).

Bash
# Execute the clinical EHR parser to generate tensors
python src/clinical_datasets/01_clinical_data_parser.py
Step 3: Run Topological Alignment
Navigate to the manifolds/ directory and execute the primary alignment notebook to generate the latent geometries and optimal transport projections.

Run manifolds/01_genomic_clinical_manifold_dann.ipynb

Step 4: Execute the Validation Suite
Navigate to the notebooks/ directory. These notebooks evaluate the models trained during the data parsing phase. They must be executed to generate the final manuscript figures:

02_feature_parsimony.ipynb

03_reliability_and_calibration.ipynb

04_temporal_degradation.ipynb

05_algorithmic_fairness.ipynb

06_loco_external_validation.ipynb

07_full_model_loco_validation.ipynb

Step 5: Final Inference
To test the multimodal late-fusion decision logic on a simulated patient:

Bash
python src/00_multimodal_inference_engine.py
License
MIT License. See LICENSE for more information.