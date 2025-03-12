import os
import argparse
import numpy as np
import h5py
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from util import get_h5_files, H5Dataset, evaluate  # Import evaluate from util.py
from tqdm import tqdm
import multiprocessing
cpu_count = multiprocessing.cpu_count()

# parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tcga_root", type=str, required=True, help="path to the tcga folder")
parser.add_argument("--cptac_root", type=str, required=False, help="path to the cptac folder")
parser.add_argument("--num_epochs", type=int, default=10, help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16384, help="batch size for dataloader")
parser.add_argument("--exp", type=str, required=True, help="path to experiment folder")
parser.add_argument("--patience", type=int, default=2, help="early stopping patience")
args = parser.parse_args()

# create experiment folder
os.makedirs(args.exp, exist_ok=True)

# save hyperparameters
with open(os.path.join(args.exp, "hyperparameters.txt"), "w") as f:
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# define the tcga folder path
tcga_root = os.path.expanduser(args.tcga_root)
luad_path = os.path.join(tcga_root, "luad")
lusc_path = os.path.join(tcga_root, "lusc")

# get luad and lusc .h5 files
luad_files = get_h5_files(luad_path)
lusc_files = get_h5_files(lusc_path)
print('File path loading done!')

# split into train-val-test
train_luad, test_luad = train_test_split(luad_files, test_size=0.2, random_state=42)
train_lusc, test_lusc = train_test_split(lusc_files, test_size=0.2, random_state=42)
train_files = train_luad + train_lusc
test_files = test_luad + test_lusc
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test
print('CV split done!')

# create dataset and dataloaders
train_dataset = H5Dataset(train_files)
val_dataset = H5Dataset(val_files)
test_dataset = H5Dataset(test_files)
num_workers = max(1, multiprocessing.cpu_count() // 4)
print(f'num_workers: {num_workers}')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
print('Datasets created!')

# define the mlp model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary classification with a single output neuron
    
    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))  # BCEWithLogitsLoss expects raw logits

print('Defining model, criterion, optim, and scheduler...')
model = MLP(1024).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print('Done!\n')

# training loop with early stopping
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []

print('Start training...')
for epoch in range(args.num_epochs):
    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    model.train()
    total_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc="Validation"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(1)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # save learning curves at each epoch
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args.exp, "learning_curve.png"))
    plt.close()
    
    # early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(args.exp, "best_model.pth"))
        print('Model saved.')
    else:
        patience_counter += 1
        print('Val loss did not improve. Skipped model saving.')
    if patience_counter >= args.patience:
        print("Early stopping triggered!")
        break

    scheduler.step()

# load the saved model. Evaluate on validation and test sets
print('Loading best model for evaluation...')
model.load_state_dict(torch.load(os.path.join(args.exp, "best_model.pth")))
model.eval()

evaluate(model, val_loader, device, "Validation", criterion)
evaluate(model, test_loader, device, "Test", criterion)

# external validation
if args.cptac_root is not None:
    cptac_root = os.path.expanduser(args.cptac_root)
    cptac_files = get_h5_files(os.path.join(cptac_root, "luad")) + get_h5_files(os.path.join(cptac_root, "lusc"))
    cptac_dataset = H5Dataset(cptac_files)
    cptac_loader = DataLoader(cptac_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    evaluate(model, cptac_loader, device, "External Validation", criterion)
