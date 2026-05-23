"""
05_feature_selection_shap.py

SHAP (SHapley Additive exPlanations) Consensus Analysis.
Trains 50 independent XGBoost classifiers on the optimal biomarker panel to 
ensure mathematical stability. Aggregates and averages the local feature 
attributions to generate a globally stable SHAP summary plot, providing 
clinical interpretability for the non-linear decision thresholds.
"""

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

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
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print(f"[*] Initiating consensus SHAP analysis ({N_ITERATIONS} iterations)...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD OPTIMAL BIOMARKER PANEL
    # ---------------------------------------------------------
    optimal_features_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not optimal_features_path.exists():
        raise FileNotFoundError(f"[ERROR] Run RFECV optimization first. Missing: {optimal_features_path.name}")
    
    optimal_genes = pd.read_csv(optimal_features_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)
    print(f"    -> Loaded optimal biomarker panel: {num_genes} features.")

    # ---------------------------------------------------------
    # 2. LOAD ISOLATED TRAINING TENSORS
    # ---------------------------------------------------------
    print("    -> Loading pre-isolated training data...")
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    
    # Slice the training tensor down to the optimal features
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
        
        # Extract SHAP values for the current iteration
        explainer = shap.TreeExplainer(model)
        accumulated_shap_values += explainer.shap_values(X_train)

    # ---------------------------------------------------------
    # 4. CALCULATE CONSENSUS IMPORTANCE
    # ---------------------------------------------------------
    print("    -> Calculating consensus SHAP metrics...")
    consensus_shap_values = accumulated_shap_values / N_ITERATIONS
    shap_importance = np.abs(consensus_shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Gene': X_train.columns,
        'Mean_Absolute_SHAP': shap_importance
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    csv_out = FEATURE_DIR / f"shap_consensus_importance_{num_genes}genes.csv"
    importance_df.to_csv(csv_out, index=False)

    # ---------------------------------------------------------
    # 5. GENERATE AESTHETIC SUMMARY PLOT
    # ---------------------------------------------------------
    print("    -> Generating clinical interpretability visualization...")
    
    aesthetic_colors = ["#4a6fe3", "#a8bdfa", "#f9f9f9", "#fca487", "#db4325"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("AestheticCmap", aesthetic_colors)

    fig = plt.figure(figsize=(11, 9))
    
    shap.summary_plot(
        consensus_shap_values, 
        X_train, 
        max_display=num_genes,  
        show=False,
        plot_size=(10, 8), 
        cmap=custom_cmap,
        alpha=0.75
    )
    
    ax = plt.gca()
    
    # Aesthetic styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#aaaaaa')
    ax.spines['bottom'].set_color('#aaaaaa')
    
    ax.tick_params(axis='both', labelsize=10, width=0.8, length=4, colors='#555555')
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12, labelpad=12, color='#444444')
    
    ax.axvline(x=0, color='#cccccc', linestyle='-', linewidth=1.0, zorder=0)
    ax.xaxis.grid(True, linestyle='-', color='#f0f0f0', alpha=0.8)
    ax.set_axisbelow(True)

    plt.title(f'SHAP Consensus Feature Importance ({num_genes} Biomarkers)', fontsize=14, pad=20, color='#333333')
    
    # Format colorbar safely
    cb = plt.gcf().axes[-1] 
    cb.tick_params(labelsize=9, colors='#555555', width=0)
    cb.set_ylabel('Gene Expression (Z-Score)', fontsize=11, labelpad=10, color='#444444')
    
    for spine in cb.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    
    pdf_out = FIG_OUT / f"Fig_SHAP_Summary_{num_genes}Genes.pdf"
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SHAP Analysis Complete. Output saved to {pdf_out.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()