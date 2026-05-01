Multimodal Sepsis Prediction Framework
This repository contains the codebase for a machine learning framework designed to evaluate sepsis outcomes by integrating high-resolution clinical time-series data with peripheral blood transcriptomic profiles.

Project Overview
Sepsis is a highly heterogeneous syndrome characterized by complex interactions between host immune responses and acute physiological decline. Predictive models have historically relied on a single data modality (either clinical data or genomic data), which can introduce observational blind spots:

Clinical-only models excel at detecting cardiovascular and respiratory collapse (e.g., changes in mean airway pressure, minimum systolic blood pressure) but often miss baseline immune dysregulation.

Genomic-only models are highly sensitive to innate immune exhaustion and neutrophil degranulation at admission, but lack real-time physiological context.

This project investigates whether a multimodal early-fusion neural network can create a comprehensive "clinical latent space" that captures the correlation between underlying immune states and acute organ failure.

Methodology
The pipeline is structured into three primary phases:

1. The Clinical Baseline
We establish a purely physiological baseline using standard ICU electronic health record (EHR) data.

Data Window: To strictly prevent target leakage, all clinical variables (vital signs, routine blood labs) are restricted to the first 24 hours of ICU admission. Patients with stays shorter than 24 hours are excluded.

Feature Engineering: Hourly time-series data is summarized into mean, min, and max values to handle data sparsity natively.

Modeling: Evaluated using sparsity-aware gradient boosting architectures (XGBoost).

2. The Genomic Baseline
We establish a parallel transcriptomic baseline to evaluate the predictive power of early-stage gene expression.

Feature Selection: Rather than utilizing full genome arrays or principal component compression, the model isolates a targeted, highly predictive 20-gene panel known to be associated with sepsis trajectory (e.g., MMP8, RETN, IL1R2, SDC4, CEACAM8).

Modeling: Evaluated using tree-based classifiers optimized for high-dimensional, low-sample-size biological data.

3. Multimodal Early Fusion Architecture (MultimodalSepsisNet)
The core of this repository is a PyTorch-based early-fusion neural network designed to integrate both modalities.

Clinical Encoder: A dense feed-forward network that compresses 24-hour physiological summaries.

Genomic Encoder: A parallel network that processes the 20-gene transcriptomic array.

Fusion Center: Both encoders project their outputs into a shared 64-dimensional latent space. The vectors are concatenated and passed through a final classification head. This architecture allows the network's gradients to jointly update both the physiological and genetic weights during backpropagation.

Repository Structure
Plaintext
multimodal_sepsis/
├── data/
│   ├── raw/                 # Raw .psv and .soft.gz files (gitignored)
│   └── processed/           # Cleaned matrices (clinical_24h_clean.csv, X_matrix.csv)
├── notebooks/
│   ├── 01_genomic_baseline.ipynb
│   ├── 02_clinical_baseline.ipynb
│   └── 03_multimodal_fusion.ipynb
├── src/
│   ├── data_parsers/        # Scripts for processing GEO and time-series EHR data
│   ├── models/              # PyTorch network class definitions
│   └── utils.py             # Evaluation metrics and SHAP plotting functions
├── Dockerfile               # Environment definition
└── README.md
Technical Stack
Environment: Linux WSL2 / Docker

Data Processing: Pandas, NumPy, Scikit-Learn

Modeling: PyTorch, XGBoost

Explainability: SHAP (SHapley Additive exPlanations)

Current Status & Roadmap
[x] Time-series clinical data parser implemented (24-hour strict window).

[x] Transcriptomic parser implemented.

[x] Unimodal baseline models trained and evaluated via SHAP.

[x] MultimodalSepsisNet architecture compiled and validated with dummy paired data.

[ ] Final evaluation on paired multi-omics / clinical ICU datasets.