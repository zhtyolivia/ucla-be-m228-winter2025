import os
import argparse
import numpy as np
import h5py
from neurocombat_sklearn import CombatModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random 

def get_h5_files(root_folder):
    h5_files = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".h5"):
                h5_files.append(os.path.join(subdir, file))
    return h5_files

def load_features_from_files(file_list):
    features, batch_labels = [], []
    for file_path in file_list:
        with h5py.File(file_path, "r") as f:
            if "features" in f:
                features.append(f["features"][:])  # load all features from this file
                batch_labels.extend([file_path] * len(f["features"]))
    return np.vstack(features), np.array(batch_labels)

# parse command-line arguments
parser = argparse.ArgumentParser(description="perform combat harmonization on tcga and cptac features")
parser.add_argument("--data_root", type=str, required=True, help="path to the root data directory")
parser.add_argument("--output_root", type=str, required=True, help="path to the output data directory")
parser.add_argument("--batch_size", type=int, default=10000, help="batch size for incremental processing")
parser.add_argument("--sample_size", type=int, default=150, help="batch size for incremental processing")
args = parser.parse_args()
os.makedirs(args.output_root, exist_ok=True)

# define the tcga folder path
tcga_root = os.path.join(args.data_root, "tcga")
luad_path = os.path.join(tcga_root, "luad")
lusc_path = os.path.join(tcga_root, "lusc")

# get luad and lusc .h5 files
luad_files = get_h5_files(luad_path)
lusc_files = get_h5_files(lusc_path)
print('file path loading done!')

# split into train-val-test (same logic as MLP training)
train_luad, test_luad = train_test_split(luad_files, test_size=0.2, random_state=42)
train_lusc, test_lusc = train_test_split(lusc_files, test_size=0.2, random_state=42)
train_files = train_luad + train_lusc
test_files = test_luad + test_lusc
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test
print('cv split done!')

# get cptac feature files
cptac_root = os.path.join(args.data_root, "cptac")
cptac_files = get_h5_files(cptac_root)

# function to load features in mini-batches
def load_features(file_list, batch_size):
    features, batch_labels = [], []
    for file_path in file_list:
        with h5py.File(file_path, "r") as f:
            if "features" in f:
                batch_size = min(batch_size, len(f["features"]))  # avoid empty batches
                features.append(f["features"][:batch_size])
                batch_labels.extend([file_path] * batch_size)
    return np.vstack(features), np.array(batch_labels)

# randomly sample h5 files for combat fitting
sampled_tcga_files = random.sample(train_files, min(len(train_files), args.sample_size // 2))
sampled_cptac_files = random.sample(cptac_files, min(len(cptac_files), args.sample_size // 2))

print(f"sampling {len(sampled_tcga_files)} tcga files and {len(sampled_cptac_files)} cptac files for combat fitting...")

# load features from the selected h5 files
sampled_tcga_features, _ = load_features_from_files(sampled_tcga_files)
sampled_cptac_features, _ = load_features_from_files(sampled_cptac_files)

# combine tcga train and cptac train data for combat
sampled_features = np.vstack([sampled_tcga_features, sampled_cptac_features])
batch_labels = np.hstack([np.zeros(len(sampled_tcga_features)), np.ones(len(sampled_cptac_features))])  # 0 for tcga, 1 for cptac

# fit combat model
print("fitting combat model...")
combat = CombatModel()
combat.fit_transform(sampled_features, batch_labels.reshape(-1, 1))
print('done!')

# function to apply combat in batches and save as new files
def apply_combat_to_h5(file_list, batch_size, combat, data_root, output_root):
    os.makedirs(output_root, exist_ok=True)  # create output directory

    for file_path in tqdm(file_list, desc="processing"):
        # preserve subdirectory structure in output path
        rel_path = os.path.relpath(file_path, data_root)
        output_path = os.path.join(output_root, rel_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # create subdirectories if needed

        with h5py.File(file_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
            if "features" not in f_in:
                print(f"warning: 'features' dataset missing in {file_path}, skipping...")
                continue

            num_samples = f_in["features"].shape[0]
            f_out.create_dataset("features", shape=f_in["features"].shape, dtype=f_in["features"].dtype)

            label = 0 if ("TCGA" in file_path or "tcga" in file_path) else 1 # TCGA=0; CPTAC=1 
            
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_features = f_in["features"][start:end, :]
                if label == 0: 
                    batch_sites = np.zeros((batch_features.shape[0], 1)) 
                elif label == 1: 
                    batch_sites = np.ones((batch_features.shape[0], 1)) 
                harmonized_features = combat.transform(batch_features, batch_sites)
                f_out["features"][start:end, :] = harmonized_features  

# define output directories
output_tcga_root = os.path.join(args.output_root, "tcga")
output_cptac_root = os.path.join(args.output_root, "cptac")

print("applying combat to tcga dataset...")
apply_combat_to_h5(train_files + val_files + test_files, args.batch_size, combat, tcga_root, output_tcga_root)

print("applying combat to cptac dataset...")
apply_combat_to_h5(cptac_files, args.batch_size, combat, cptac_root, output_cptac_root)

print("combat harmonization completed successfully!")