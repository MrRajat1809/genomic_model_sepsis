"""
05_loco_meta_analysis.py

Leave-One-Cohort-Out (LOCO) Meta-Analysis.
Iteratively trains the XGBoost model on N-1 cohorts and tests on the held-out cohort 
(strictly within the Mega-Train bounds; Vault is excluded). 
Computes 1,000-iteration bootstrapped 95% CIs for each cohort, calculates 
Meta-Analysis Heterogeneity (Cochran's Q and I^2), and generates a native Forest Plot.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

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
PLOT_DATA_DIR = BASE_DIR / "outputs" / "plot_data"

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
    print("==================================================")
    print("[*] STARTING ARMORED LOCO VALIDATION (META-ANALYSIS)")
    print("==================================================")
    
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD FEATURES & TENSORS
    # ---------------------------------------------------------
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"[ERROR] Optimal features missing: {feature_path.name}")
        
    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)
    print(f"    -> Target Signature: {num_genes} genes.")

    # Strictly load the pre-isolated Training Tensors
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    meta_train_df = pd.read_csv(DATA_DIR / "meta_train.csv")
    
    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    
    X_train_master = X_train_full[optimal_genes]
    y_train_master = y_train_df[target_col].astype(int)

    training_cohorts = meta_train_df['Dataset'].unique()
    
    forest_records = []
    roc_records = []

    # ---------------------------------------------------------
    # 2. EXECUTE LEAVE-ONE-COHORT-OUT (LOCO) LOOP
    # ---------------------------------------------------------
    print("\n    -> Executing LOCO Iterations & Bootstrapping (1,000 resamples)...")
    
    for held_out_cohort in training_cohorts:
        # Partition data
        test_mask = meta_train_df['Dataset'] == held_out_cohort
        train_mask = ~test_mask
        
        X_train = X_train_master[train_mask]
        y_train = y_train_master[train_mask]
        X_test = X_train_master[test_mask]
        y_test = y_train_master[test_mask]
        
        if len(y_test.unique()) < 2:
            print(f"       [SKIP] {held_out_cohort} lacks mortality variation.")
            continue
            
        scale_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9))
        
        loco_model = SklearnCompatibleXGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4,
            scale_pos_weight=scale_weight, 
            eval_metric='logloss', 
            objective='binary:logistic',
            random_state=42, 
            n_jobs=-1
        )
        
        loco_model.fit(X_train, y_train)
        probs = loco_model.predict_proba(X_test)[:, 1]
        
        base_auc = roc_auc_score(y_test, probs)
        
        # ---------------------------------------------------------
        # 3. STATISTICAL ARMOR: BOOTSTRAPPING CIs
        # ---------------------------------------------------------
        n_bootstraps = 1000
        rng = np.random.RandomState(42)
        boot_aucs = []
        y_test_arr = y_test.values
        
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_test_arr), len(y_test_arr))
            if len(np.unique(y_test_arr[indices])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_test_arr[indices], probs[indices]))
            
        auc_lower = np.percentile(boot_aucs, 2.5)
        auc_upper = np.percentile(boot_aucs, 97.5)
        auc_se = np.std(boot_aucs)  
        
        print(f"       -> Held Out: {held_out_cohort:<10} | AUC: {base_auc:.3f} (95% CI: [{auc_lower:.3f}, {auc_upper:.3f}])")
        
        forest_records.append({
            "Cohort": held_out_cohort,
            "N_Patients": len(y_test),
            "AUC": base_auc,
            "AUC_Lower": auc_lower,
            "AUC_Upper": auc_upper,
            "SE": auc_se
        })
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        for f, t in zip(fpr, tpr):
            roc_records.append({"Cohort": held_out_cohort, "FPR": f, "TPR": t})

    # ---------------------------------------------------------
    # 4. META-ANALYSIS (INVERSE VARIANCE WEIGHTING)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("[*] COMPUTING META-ANALYSIS STATISTICS")
    print("="*50)

    df_forest = pd.DataFrame(forest_records)

    # Inverse Variance Weights
    df_forest['Weight'] = 1 / (df_forest['SE'] ** 2)
    sum_weights = df_forest['Weight'].sum()

    # Weighted Mean AUC
    weighted_mean_auc = (df_forest['AUC'] * df_forest['Weight']).sum() / sum_weights

    # Cochran's Q and I^2 Heterogeneity
    df_forest['Q_contrib'] = df_forest['Weight'] * ((df_forest['AUC'] - weighted_mean_auc) ** 2)
    Q = df_forest['Q_contrib'].sum()
    k = len(df_forest)
    df_stat = k - 1
    I2 = max(0.0, 100 * (Q - df_stat) / Q)

    se_pooled = 1 / np.sqrt(sum_weights)
    pooled_lower = weighted_mean_auc - (1.96 * se_pooled)
    pooled_upper = weighted_mean_auc + (1.96 * se_pooled)

    print(f"    -> Pooled LOCO AUC:  {weighted_mean_auc:.3f} (95% CI: [{pooled_lower:.3f}, {pooled_upper:.3f}])")
    print(f"    -> Cochran's Q:      {Q:.2f} (df={df_stat})")
    print(f"    -> I^2 (Dispersion): {I2:.1f}%")
    
    if I2 > 50:
        print("    -> [!] Moderate/High Heterogeneity Detected.")
    else:
        print("    -> [+] Low Heterogeneity: The model behaves consistently across hospital systems.")

    # ---------------------------------------------------------
    # 5. GENERATE NATIVE FOREST PLOT
    # ---------------------------------------------------------
    print("\n    -> Generating Aesthetic Forest Plot...")
    
    df_forest = df_forest.sort_values(by='AUC', ascending=True).reset_index(drop=True)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#aaaaaa"})
    
    y_pos = np.arange(len(df_forest))
    
    # Plot individual cohort AUROCs
    plt.errorbar(
        df_forest['AUC'], y_pos, 
        xerr=[df_forest['AUC'] - df_forest['AUC_Lower'], df_forest['AUC_Upper'] - df_forest['AUC']],
        fmt='s', color='#4a6fe3', ecolor='#aaaaaa', elinewidth=2, capsize=4, markersize=8
    )
    
    # Y-axis labels
    plt.yticks(y_pos, [f"{row['Cohort']} (n={int(row['N_Patients'])})" for _, row in df_forest.iterrows()], fontsize=11)
    
    # Text annotations
    for i, row in df_forest.iterrows():
        text_label = f"{row['AUC']:.2f} [{row['AUC_Lower']:.2f}-{row['AUC_Upper']:.2f}]"
        plt.text(1.05, i, text_label, va='center', fontsize=10, color='#333333')

    # Pooled Estimate Marker
    plt.axvline(x=weighted_mean_auc, color='#db4325', linestyle='--', linewidth=2, label=f'Pooled AUC: {weighted_mean_auc:.2f}')
    plt.axvspan(pooled_lower, pooled_upper, color='#db4325', alpha=0.15)

    plt.xlim(0.4, 1.0)
    plt.xlabel('Area Under the ROC Curve (AUROC)', fontsize=12, labelpad=10, fontweight='bold')
    plt.title(f"Meta-Analysis: LOCO Validation of the {num_genes}-Gene Signature\nHeterogeneity ($I^2$): {I2:.1f}%", fontsize=14, pad=15)
    plt.legend(loc='lower left', frameon=True)
    
    # Clean spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Export
    plot_pdf = FIG_OUT / f"Fig_LOCO_Forest_Plot_{num_genes}Genes.pdf"
    forest_csv = PLOT_DATA_DIR / f"loco_forest_data_{num_genes}genes.csv"
    roc_csv = PLOT_DATA_DIR / f"loco_roc_curves_{num_genes}genes.csv"

    plt.savefig(plot_pdf, dpi=300, bbox_inches='tight')
    df_forest.to_csv(forest_csv, index=False)
    pd.DataFrame(roc_records).to_csv(roc_csv, index=False)

    print(f"[*] Export Complete! Forest Plot saved to {plot_pdf.name}")

if __name__ == "__main__":
    main()