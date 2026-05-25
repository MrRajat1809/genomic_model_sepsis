"""
05_feature_selection_shap.py

SHAP Consensus Analysis Computation.
Trains 50 independent XGBoost classifiers on the optimal biomarker panel.
Aggregates and averages the local feature attributions to generate a 
globally stable SHAP matrix, saving the raw values to disk for downstream plotting.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"

N_ITERATIONS = 50  

# ==========================================
# COMPATIBILITY WRAPPER
# ==========================================
class SklearnCompatibleXGBClassifier(xgb.XGBClassifier):
    """Bypasses Scikit-Learn 1.6.0+ tag delegation limitations."""
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if hasattr(tags, "estimator_type"):
            tags.estimator_type = "classifier"
        elif isinstance(tags, dict):
            tags["estimator_type"] = "classifier"
        return tags

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print(f"[*] Initiating consensus SHAP analysis ({N_ITERATIONS} iterations)...")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD OPTIMAL BIOMARKER PANEL
    # ---------------------------------------------------------
    optimal_features_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not optimal_features_path.exists():
        print(f"[ERROR] Run RFECV optimization first. Missing: {optimal_features_path.name}")
        return
        
    optimal_genes = pd.read_csv(optimal_features_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)

    # ---------------------------------------------------------
    # 2. LOAD ISOLATED TRAINING TENSORS
    # ---------------------------------------------------------
    print("    -> Loading pre-isolated training data...")
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    
    X_train = X_train_full[optimal_genes]
    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    y_train = y_train_df[target_col].astype(int)

    # ---------------------------------------------------------
    # 3. ITERATIVE MODEL TRAINING & SHAP EXTRACTION
    # ---------------------------------------------------------
    print("    -> Executing iterative model training and SHAP extraction...")
    scale_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9))
    accumulated_shap_values = np.zeros(X_train.shape)

    for i in range(N_ITERATIONS):
        model = SklearnCompatibleXGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4, 
            scale_pos_weight=scale_weight, 
            eval_metric='logloss',
            objective='binary:logistic',
            random_state=i, 
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(model)
        accumulated_shap_values += explainer.shap_values(X_train)

    # ---------------------------------------------------------
    # 4. CALCULATE & EXPORT CONSENSUS METRICS
    # ---------------------------------------------------------
    print("    -> Exporting consensus SHAP metrics and raw matrices...")
    consensus_shap_values = accumulated_shap_values / N_ITERATIONS
    
    # Save the raw numpy array required for the SHAP summary plot
    np.save(FEATURE_DIR / "raw_consensus_shap_values.npy", consensus_shap_values)

    # Save the readable CSV for tabular reporting
    shap_importance = np.abs(consensus_shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Gene': X_train.columns,
        'Mean_Absolute_SHAP': shap_importance
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    csv_out = FEATURE_DIR / f"shap_consensus_importance_{num_genes}genes.csv"
    importance_df.to_csv(csv_out, index=False)

    print(f"[*] SUCCESS! SHAP data saved to {FEATURE_DIR.name}/")

if __name__ == "__main__":
    main()