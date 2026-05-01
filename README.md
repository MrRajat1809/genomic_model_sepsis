# Multimodal Sepsis Prediction Framework 

This repository contains the codebase for a machine learning framework designed to evaluate sepsis outcomes by integrating high-resolution clinical time-series data with peripheral blood transcriptomic profiles.

---

## Project Overview

Sepsis is a highly heterogeneous syndrome characterized by complex interactions between host immune responses and acute physiological decline. Predictive models have historically relied on a single data modality (either clinical data or genomic data), which can introduce observational blind spots.

**Clinical-only models** excel at detecting cardiovascular and respiratory collapse (e.g., changes in mean airway pressure, minimum systolic blood pressure) but often miss baseline immune dysregulation.

**Genomic-only models** are highly sensitive to innate immune exhaustion and neutrophil degranulation at admission but lack real-time physiological context.

This project investigates whether a **multimodal early-fusion neural network** can create a comprehensive *clinical latent space* that captures the correlation between underlying immune states and acute organ failure.

---

## Methodology

The pipeline is structured into three primary phases:

---

### 1. Clinical Baseline

We establish a purely physiological baseline using standard ICU electronic health record (EHR) data.

#### Data Window

To strictly prevent target leakage, all clinical variables (vital signs, routine blood labs) are restricted to the **first 24 hours of ICU admission**. Patients with stays shorter than 24 hours are excluded.

#### Feature Engineering

Hourly time-series data is summarized into:

* Mean
* Minimum
* Maximum

This approach handles data sparsity while preserving clinically relevant signal dynamics.

#### Modeling

* Algorithm: **XGBoost**
* Rationale: Robust performance on structured clinical data with missingness
* Design: Sparsity-aware gradient boosting architecture

---

### 2. Genomic Baseline

We establish a parallel transcriptomic baseline to evaluate the predictive power of early-stage gene expression.

#### Feature Selection

Rather than utilizing full genome arrays or principal component compression, the model isolates a targeted, biologically grounded **20-gene panel** associated with sepsis trajectory.

**Example genes include:**

* MMP8
* RETN
* IL1R2
* SDC4
* CEACAM8

This targeted approach improves interpretability and reduces dimensional instability.

#### Modeling

* Algorithm: Tree-based classifiers
* Optimization target: High-dimensional, low-sample-size biological datasets
* Evaluation: Cross-validation with SHAP-based interpretability

---

### 3. Multimodal Early Fusion Architecture

## MultimodalSepsisNet

The core component of this repository is a **PyTorch-based early-fusion neural network** designed to integrate physiological and transcriptomic data into a unified predictive representation.

---

#### Clinical Encoder

A dense feed-forward network that compresses summarized physiological features from the first 24 hours of ICU admission.

---

#### Genomic Encoder

A parallel neural network that processes the 20-gene transcriptomic expression array.

---

#### Fusion Center

Both encoders project their outputs into a shared **64-dimensional latent space**.

The latent vectors are:

1. Concatenated
2. Passed through a final classification head
3. Optimized jointly via backpropagation

This architecture enables gradient-driven coordination between physiological and genomic representations, allowing the model to learn cross-modal relationships between immune state and organ dysfunction.

---

## Repository Structure

```plaintext
multimodal_sepsis/
│
├── data/
│   ├── raw/                 # Raw .psv and .soft.gz files (gitignored)
│   └── processed/           # Cleaned matrices
│       ├── clinical_24h_clean.csv
│       └── X_matrix.csv
│
├── notebooks/
│   ├── 01_genomic_baseline.ipynb
│   ├── 02_clinical_baseline.ipynb
│   └── 03_multimodal_fusion.ipynb
│
├── src/
│   ├── data_parsers/        # GEO and time-series EHR processing scripts
│   ├── models/              # PyTorch network definitions
│   └── utils.py             # Metrics, evaluation, SHAP visualization
│
├── Dockerfile               # Reproducible environment definition
└── README.md
```

---

## Technical Stack

### Environment

* Linux (WSL2)
* Docker

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Modeling

* PyTorch
* XGBoost

### Explainability

* SHAP (SHapley Additive exPlanations)

---

## Current Status

```text
[x] Time-series clinical data parser implemented (24-hour strict window)
[x] Transcriptomic parser implemented
[x] Unimodal baseline models trained and evaluated using SHAP
[x] MultimodalSepsisNet architecture compiled and validated with dummy paired data
[ ] Final evaluation on paired multi-omics and clinical ICU datasets
```

---

## Research Objective

The primary objective of this framework is to evaluate whether integrating physiological and transcriptomic modalities through early-fusion deep learning can improve early risk stratification in sepsis compared to unimodal predictive models.

---

## Reproducibility

The repository includes:

* Containerized environment definition (Docker)
* Deterministic preprocessing pipeline
* Modular model architecture
* Explainability tooling via SHAP

This design supports:

* Reproducible experimentation
* External validation
* Clinical machine learning transparency
* Peer-reviewed publication readiness

---

## License

Specify license here (e.g., MIT, Apache 2.0, or research-use license).
