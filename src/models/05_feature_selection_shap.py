"""
05_feature_selection_shap.py

Trains 50 independent XGBoost classifiers on the 26 selected genes.
Aggregates and averages the SHAP values, then outputs the summary plot.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
PLOT_DATA_DIR = BASE_DIR / "outputs" / "plot_data"
FIG_OUT = BASE_DIR / "outputs" / "figures"

HOLDOUT_COHORT = 'GSE65682'
N_ITERATIONS = 50  

# ==========================================
# XGBOOST TAG FIX
# ==========================================
class SklearnCompatibleXGBClassifier(xgb.XGBClassifier):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if hasattr(tags, "estimator_type"):
            tags.estimator_type = "classifier"
        elif isinstance(tags, dict):
            tags["estimator_type"] = "classifier"
        return tags

def main():
    print(f"[*] RUNNING SHAP ANALYSIS ({N_ITERATIONS} ITERATIONS)...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the 26 Genes
    optimal_features_path = PLOT_DATA_DIR / "03_optimal_feature_list.csv"
    if not optimal_features_path.exists():
        raise FileNotFoundError("Run 04_rfecv_optimization.py first!")
    
    optimal_genes = pd.read_csv(optimal_features_path)['Optimal_Genes'].tolist()

    # 2. Load Tensors
    X_elite = pd.read_csv(DEG_DIR / "X_deg_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")
    
    X_optimal = X_elite[optimal_genes]
    target_col = 'Mortality' if 'Mortality' in y.columns else y.columns[0]

    # 3. Isolate Training Cohort
    train_mask = meta['Dataset'] != HOLDOUT_COHORT
    X_train = X_optimal[train_mask]
    y_train = y[train_mask][target_col].astype(int)

    # 4. Iterative Training
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

    # 5. Calculate Mean SHAP Values
    consensus_shap_values = accumulated_shap_values / N_ITERATIONS

    # 6. Extract Feature Importance
    shap_importance = np.abs(consensus_shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'Gene': X_train.columns,
        'Mean_Absolute_SHAP': shap_importance
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)

    importance_df.to_csv(DEG_DIR / "SHAP_Consensus_Feature_Importance_26Genes.csv", index=False)

    # 7. Generate Aesthetic SHAP Summary Plot
    aesthetic_colors = ["#4a6fe3", "#a8bdfa", "#f9f9f9", "#fca487", "#db4325"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("AestheticCmap", aesthetic_colors)

    fig = plt.figure(figsize=(11, 9))
    
    shap.summary_plot(
        consensus_shap_values, 
        X_train, 
        max_display=26,  
        show=False,
        plot_size=(10, 8), 
        cmap=custom_cmap,
        alpha=0.75
    )
    
    ax = plt.gca()
    
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

    plt.title('SHAP Feature Importance (26 Features)', fontsize=14, pad=20, color='#333333')
    
    cb = plt.gcf().axes[-1] 
    cb.tick_params(labelsize=9, colors='#555555', width=0)
    cb.set_ylabel('Feature Value', fontsize=11, labelpad=10, color='#444444')
    
    # [THE FIX]: Safely remove colorbar borders directly from the axes spines
    for spine in cb.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    
    pdf_out = FIG_OUT / "Fig5_SHAP_26Genes.pdf"
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Figure saved to {pdf_out.name}")

if __name__ == "__main__":
    main()