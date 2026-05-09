"""
01_model_benchmark.py

Algorithmic Benchmarking for Sepsis Mortality Prediction.
Evaluates multiple machine learning classifiers (linear, non-linear, and ensemble)
on the processed transcriptomic/clinical tensors. Compares models based on 
ROC-AUC, Accuracy, and training time, outputting a consolidated results table 
for manuscript reporting.
"""

import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    AdaBoostClassifier
)
import xgboost as xgb

# Suppress sklearn/xgboost warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/models/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
OUT_DIR = BASE_DIR / "outputs" / "models"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING ALGORITHMIC BENCHMARKING...")

    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate input data exists
    x_path = DATA_DIR / "X_master.csv.gz"
    y_path = DATA_DIR / "y_master.csv"
    
    if not x_path.exists() or not y_path.exists():
        print(f"[!] Critical Error: Tensors not found in {DATA_DIR.name}.")
        print("[!] Please run the tensor merger script first.")
        return

    # 1. Load Data
    print("    -> Loading Master Tensors...")
    X = pd.read_csv(x_path, compression='gzip')
    y = pd.read_csv(y_path)

    # 2. Stratified Split
    print("    -> Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y['Mortality'], test_size=0.2, stratify=y['Mortality'], random_state=42
    )

    # 3. Standardize Features 
    # Vital for Logistic Regression, SVM, and KNN. 
    # Fit ONLY on training data to prevent data leakage.
    print("    -> Scaling features (Z-score normalization)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Calculate weight for imbalanced datasets
    scale_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

    # 4. Define the Benchmark Dictionary
    # Using 'balanced' class weights where applicable to handle mortality rate imbalance
    classifiers = {
        "Naive Bayes (Gaussian)": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression (L2)": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Support Vector Machine (RBF)": SVC(probability=True, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, scale_pos_weight=scale_weight, eval_metric='auc', random_state=42, n_jobs=-1)
    }

    # 5. Execute the Benchmark
    results = []
    print("\n[*] TRAINING MODELS (This will take a few minutes)...")
    print("-" * 65)
    print(f"{'Algorithm':<30} | {'ROC-AUC':<10} | {'Accuracy':<10} | {'Time (s)':<10}")
    print("-" * 65)

    for name, clf in classifiers.items():
        start_time = time.time()
        
        try:
            # Train
            clf.fit(X_train_scaled, y_train)
            
            # Predict Probabilities for AUC
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test_scaled)[:, 1]
            else:
                # Fallback if probability is disabled
                y_prob = clf.decision_function(X_test_scaled) 
                
            # Predict Classes for Accuracy
            y_pred = clf.predict(X_test_scaled)
            
            # Metrics
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            
        except Exception as e:
            print(f"{name:<30} | ERROR: {e}")
            continue
            
        elapsed = time.time() - start_time
        
        # Print real-time results
        print(f"{name:<30} | {auc:.4f}     | {acc:.4f}     | {elapsed:.2f}")
        
        results.append({
            "Algorithm": name,
            "ROC-AUC": round(auc, 4),
            "Accuracy": round(acc, 4),
            "Time_Seconds": round(elapsed, 2)
        })

    print("-" * 65)

    # 6. Save results for the manuscript
    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    csv_path = OUT_DIR / "algorithm_benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"\n[*] SUCCESS! Benchmark table saved for manuscript to:")
    print(f"    -> {csv_path}")

if __name__ == "__main__":
    main()