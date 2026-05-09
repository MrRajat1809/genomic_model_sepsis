"""
02_xgboost_baseline.py

The "Molecular Geneticist" Model.
Trains a robust XGBoost baseline classifier on the high-dimensional transcriptomic
feature space (HUGO gene symbols) to predict sepsis mortality. Extracts global 
feature importances to identify top mortality-associated biomarkers, and saves 
the trained weights for downstream meta-model integration.
"""

import warnings
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/models/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
MODEL_OUT = BASE_DIR / "outputs" / "models"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING 'MOLECULAR GENETICIST' XGBOOST BASELINE...")

    # Force Python to build the folders safely
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    x_path = DATA_DIR / "X_master.csv.gz"
    y_path = DATA_DIR / "y_master.csv"

    if not x_path.exists() or not y_path.exists():
        print(f"[!] Critical Error: Tensors not found in {DATA_DIR.name}.")
        print("[!] Please run the tensor merger script first.")
        return

    print("    -> Loading Master Tensors (This will take a moment)...")
    X = pd.read_csv(x_path, compression='gzip')
    y = pd.read_csv(y_path)

    print(f"    -> Data loaded. X shape: {X.shape}, y shape: {y.shape}")

    # 2. Check Class Imbalance
    # Sepsis datasets usually have more survivors (0) than non-survivors (1)
    mortality_rate = y['Mortality'].mean() * 100
    print(f"    -> Mortality Rate in dataset: {mortality_rate:.1f}%")

    # 3. Stratified Split
    # Stratified ensures the 80/20 split keeps the exact same mortality ratio in both sets
    print("    -> Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y['Mortality'], test_size=0.2, stratify=y['Mortality'], random_state=42
    )

    # Calculate dynamic weight to handle class imbalance
    scale_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

    # 4. Initialize XGBoost
    print("    -> Initializing XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,              # Build 300 sequential trees
        learning_rate=0.05,            # Learn slowly to prevent overfitting
        max_depth=5,                   # Restrict tree depth to handle noise
        subsample=0.8,                 # Use 80% of patients per tree
        colsample_bytree=0.8,          # Use 80% of genes per tree
        scale_pos_weight=scale_weight, # Automatically balance the classes
        eval_metric='auc',             # Area Under the Curve (Clinical Standard)
        random_state=42,
        n_jobs=-1                      # Use all available CPU cores
    )

    # 5. Train
    print("    -> Training Baseline Model (Hold on to your CPU)...")
    model.fit(X_train, y_train)

    # 6. Evaluate
    print("    -> Generating Predictions...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get raw probabilities for AUC

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*50)
    print("[*] BASELINE PERFORMANCE (Molecular Geneticist)")
    print(f"    -> Accuracy: {acc:.4f}")
    print(f"    -> ROC-AUC:  {auc:.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Extract Biology (Feature Importance)
    print("    -> Extracting Top 20 Biological Biomarkers...")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Gene': X.columns, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)

    # 8. Plot and Save
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Gene', data=fi_df, palette='magma')
    plt.title('Top 20 Sepsis Mortality Biomarkers (Molecular Geneticist)')
    plt.xlabel('Relative Importance Score')
    plt.ylabel('HUGO Gene Symbol')
    plt.tight_layout()
    
    # Save with the new naming convention
    fig_path = FIG_OUT / "molecular_geneticist_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    model_path = MODEL_OUT / "molecular_geneticist_baseline.json"
    model.save_model(model_path)
    
    print(f"\n[*] Baseline Model saved to: {model_path}")
    print(f"[*] Biomarker Chart saved to: {fig_path}")

if __name__ == "__main__":
    main()