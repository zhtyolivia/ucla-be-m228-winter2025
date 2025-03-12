import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score

class H5Dataset(Dataset):
    def __init__(self, file_list):
        self.tile_index = []  # Stores (file_path, tile_idx)
        self.file_list = file_list
        for file_path in self.file_list:
            with h5py.File(file_path, "r") as h5_file:
                if "features" in h5_file:
                    num_tiles = h5_file["features"].shape[0]
                    self.tile_index.extend([(file_path, i) for i in range(num_tiles)])


    def __len__(self):
        return len(self.tile_index)  # Number of total tiles, not files

    def __getitem__(self, idx):
        file_path, tile_idx = self.tile_index[idx]  # Get the correct file and tile index
        with h5py.File(file_path, "r") as h5_file:
            features = h5_file["features"][tile_idx]  # Load only one tile at a time
        label = 0 if ("LUAD" in file_path or "luad" in file_path) else 1  
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_features(file_list):
    """Load all features and labels from a list of .h5 files into memory."""
    features_list, labels_list = [], []
    for file_path in file_list:
        with h5py.File(file_path, "r") as h5_file:
            if "features" in h5_file:
                features = h5_file["features"][:]
                label = 0 if ("LUAD" in file_path or "luad" in file_path) else 1
                labels = np.full(features.shape[0], label)
                features_list.append(features)
                labels_list.append(labels)
    return np.vstack(features_list), np.concatenate(labels_list)


# function to get all .h5 files recursively
def get_h5_files(root_folder):
    h5_files = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(subdir, file))
    return h5_files

# function to load features one file at a time
def stream_features(file_list):
    for file_path in file_list:
        with h5py.File(file_path, "r") as h5_file:
            if "features" in h5_file:
                features = h5_file["features"][:]
                label = 0 if "LUAD" in file_path else 1
                labels = np.full(features.shape[0], label)
                yield features, labels


def evaluate(model, loader, device, name, criterion):
    model.eval()
    total_loss = 0
    y_true, y_pred_probs, y_pred_labels = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc=f"Evaluating {name}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)  # Convert probabilities to binary labels
            
            y_true.extend(y_batch.cpu().numpy().flatten())
            y_pred_probs.extend(probs)
            y_pred_labels.extend(preds)
    
    # Compute loss
    avg_loss = total_loss / len(loader)
    
    # Compute classification metrics
    auroc = roc_auc_score(y_true, y_pred_probs)
    auprc = average_precision_score(y_true, y_pred_probs)
    acc = accuracy_score(y_true, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average="binary", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_labels)
    
    # Print results
    print(f"\n{name} Evaluation:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}\n")

    return {
        "loss": avg_loss,
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc
    }