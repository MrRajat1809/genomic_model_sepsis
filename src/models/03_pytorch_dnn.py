import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING DEEP LEARNING GENETICIST (PyTorch)...")

# 1. Hardware Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    -> Hardware detected: {device.type.upper()}")

base_dir = "/workspace"
data_dir = os.path.join(base_dir, "data/processed/ml_tensors")
out_dir = os.path.join(base_dir, "outputs/models")
os.makedirs(out_dir, exist_ok=True)

# 2. PyTorch Dataset Class
class TranscriptomicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1) # Reshape for BCE loss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Deep Neural Network Architecture
class DeepGeneticist(nn.Module):
    def __init__(self, input_dim):
        super(DeepGeneticist, self).__init__()
        
        # Funnel architecture: 7902 -> 1024 -> 256 -> 64 -> 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5), # Aggressive dropout to prevent overfitting
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)
            # No Sigmoid here because we use BCEWithLogitsLoss (more numerically stable)
        )

    def forward(self, x):
        return self.network(x)

# 4. Load & Prep Data
print("    -> Loading Master Tensors...")
X = pd.read_csv(os.path.join(data_dir, "X_master.csv.gz"), compression='gzip')
y = pd.read_csv(os.path.join(data_dir, "y_master.csv"))

input_dim = X.shape[1]

# Stratified Split & Scale
print("    -> Splitting and Scaling data...")
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y['Mortality'].values, test_size=0.2, stratify=y['Mortality'], random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weight for PyTorch
pos_weight_val = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

# Build DataLoaders
train_dataset = TranscriptomicDataset(X_train_scaled, y_train)
test_dataset = TranscriptomicDataset(X_test_scaled, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 5. Initialize Model, Loss, and Optimizer
model = DeepGeneticist(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Handles class imbalance
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3) # L2 Regularization

# 6. Training Loop
EPOCHS = 40
print(f"\n[*] TRAINING NEURAL NETWORK ({EPOCHS} Epochs)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Quick validation check every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(batch_y.numpy())
                
        val_auc = roc_auc_score(val_targets, val_probs)
        print(f"    -> Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {total_loss/len(train_loader):.4f} | Val ROC-AUC: {val_auc:.4f}")

# 7. Final Evaluation
print("\n[*] GENERATING FINAL PREDICTIONS...")
model.eval()
final_probs = []
final_preds = []
y_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        logits = model(batch_X)
        probs = torch.sigmoid(logits).cpu().numpy()
        
        final_probs.extend(probs)
        final_preds.extend((probs >= 0.5).astype(int))
        y_true.extend(batch_y.numpy())

acc = accuracy_score(y_true, final_preds)
auc = roc_auc_score(y_true, final_probs)

print("\n" + "="*50)
print("[*] DEEP LEARNING PERFORMANCE (PyTorch DNN)")
print(f"    -> Accuracy: {acc:.4f}")
print(f"    -> ROC-AUC:  {auc:.4f}")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_true, final_preds))

# Save the brain
torch.save(model.state_dict(), os.path.join(out_dir, "deep_geneticist_weights.pth"))
print(f"[*] Deep Geneticist saved to outputs/models/")