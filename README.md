# Multimodal Sepsis Prediction Framework

This repository provides a pipeline for evaluating sepsis outcomes by integrating clinical time-series data with peripheral blood transcriptomic profiles. The framework compares unimodal baselines against a combined multimodal architecture to assess predictive performance.

---

## Project Overview

Sepsis is characterized by complex interactions between host immune responses and physiological decline. This project evaluates predictive models using two distinct data modalities:

* **Clinical Data:** Electronic Health Record (EHR) data focusing on cardiovascular and respiratory indicators.
* **Genomic Data:** Transcriptomic profiles focusing on early-stage innate immune markers.

The framework utilizes a **multimodal late-fusion approach** (and experimental early-fusion deep learning) to investigate whether combining these features improves risk stratification compared to isolated models.

---

## Methodology

The pipeline consists of three phases:

### 1. Clinical Baseline (`ICU Doctor`)

A physiological baseline developed from the PhysioNet 2019 Challenge dataset (~40,000 patients).

#### Data Window

Variable-length ICU stays summarized into patient-level features.

#### Feature Engineering

* 104 clinical variables (vitals and labs)
* Summary statistics:

  * Mean
  * Minimum
  * Maximum

#### Model

* **XGBoost**

#### Rationale

Gradient boosting is utilized for its native handling of missing data (Sparsity-Aware Split Finding) and robust performance on tabular ICU data.

#### Performance

Achieved an ROC-AUC of **0.88** on sepsis onset prediction.

---

### 2. Genomic Baseline (`The Geneticist`)

A transcriptomic baseline developed from curated Gene Expression Omnibus (GEO) datasets.

#### Feature Selection

High-dimensional input (**7,902 genes**) processed without initial dimensionality reduction to allow the model to identify non-linear interactions.

#### Model

* **XGBoost**

#### Rationale

Benchmarking against eight classifiers (including Naive Bayes, K-Nearest Neighbors, and Support Vector Machines) identified tree ensembles as the optimal architecture for this tabular biological data.

#### Performance

Achieved an ROC-AUC of **0.76** on sepsis mortality prediction.

---

### 3. Multimodal Fusion Engine

An inference system designed to integrate the outputs of the two specialized models.

#### Architecture

Late-fusion decision logic.

#### Mechanism

The system generates independent probabilities for:

* **Sepsis Onset** (Clinical Model)
* **Mortality Risk** (Genomic Model)

#### Output

A unified **ICU Dashboard** that categorizes patient risk into four tiers based on cross-modal thresholds:

1. Low Risk
2. Monitoring
3. Warning
4. Code Red

---

## Repository Structure

```plaintext
multimodal_sepsis/
│
├── data/
│   ├── raw/                 # Raw .psv and GEO files (gitignored)
│   └── processed/           # Processed tensors
│       ├── clinical_master_raw.csv.gz
│       └── X_master.csv.gz
│
├── src/
│   ├── clinical_datasets/   # EHR parsing and XGBoost baseline scripts
│   ├── models/              # Model benchmarking and PyTorch DNN experiments
│   ├── 00_multimodal_inference_engine.py  # Final fusion script
│   └── 01_shap_analysis.py  # Explainability scripts
│
├── outputs/
│   ├── models/              # Saved model weights (.json and .pth)
│   └── figures/             # SHAP and Feature Importance visualizations
│
└── Dockerfile               # Environment definition
```

---

## Technical Stack

### Modeling

* XGBoost
* PyTorch
* Scikit-learn

### Data Analysis

* Pandas
* NumPy

### Explainability

**SHAP (SHapley Additive exPlanations)** used to validate:

* Clinical logic (e.g., Lactate and FiO2 importance)
* Biological drivers (e.g., PCSK9 and RORC)

### Environment

Docker-containerized for reproducibility with NVIDIA GPU support (CUDA).

---

## Key Findings

1. **Algorithmic Selection**
   Standard Deep Learning (MLP) architectures underperformed relative to XGBoost on genomic tabular data (ROC-AUC **0.65 vs 0.76**) due to high dimensionality and small sample sizes.

2. **Clinical Features**
   SHAP analysis confirmed that the model prioritized clinically recognized markers of septic shock, including:

   * Maximum temperature
   * High FiO2 (ventilator dependence)
   * Elevated Lactate

3. **Genomic Markers**
   The model identified **PCSK9** expression as a primary driver of mortality risk, aligning with current research regarding endotoxin clearance.

---

## Reproducibility

To replicate the environment and run the inference engine:

```bash
# Build the container
 docker-compose build

# Run the clinical parser
 python src/clinical_datasets/01_clinical_data_parser.py

# Execute the fusion engine
 python src/00_multimodal_inference_engine.py
```

---

## License

MIT License
