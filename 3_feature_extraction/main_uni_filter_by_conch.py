import sys 
import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import timm
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np
import pandas as pd 

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path

class Filtered_Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        conch_preds_path,
        img_transforms=None,
        print_summary=False):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            wsi (OpenSlide object): Whole slide image object.
            conch_preds_path (string): Path to CSV file containing model predictions.
            img_transforms (callable, optional): Optional transform to be applied on a sample.
            print_summary (bool, optional): Whether to print summary info.
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path
        self.conch_preds_path = conch_preds_path 

        # Load CSV predictions and filter patches with pred_label > 0
        self.conch_df = pd.read_csv(self.conch_preds_path)
        self.filtered_conch_df = self.conch_df[self.conch_df['pred_label'] > 0]  # Keep only LUAD/LUSC
        
        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            
            # Read coordinates from H5
            all_coords = np.array(dset)
            
            # Match coordinates with filtered predictions
            filtered_coords = []
            for coord in self.filtered_conch_df[['x', 'y']].values:
                mask = (all_coords == coord).all(axis=1)  # Find matching row
                if mask.any():
                    filtered_coords.append(coord)
            
            self.filtered_coords = np.array(filtered_coords)
            self.length = len(self.filtered_coords)  # Update dataset length to filtered size

        if print_summary: 
            self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        """Prints summary of dataset and feature extraction settings."""
        print("Dataset Summary:")
        print(f"Original patches in H5: {len(self.conch_df)}")
        print(f"Filtered patches (LUAD/LUSC only): {self.length}")
        print(f"Patch Level: {self.patch_level}")
        print(f"Patch Size: {self.patch_size}")
        print("Transformations:", self.roi_transforms)

    def __getitem__(self, idx):
        """Fetches filtered patch based on coordinates."""
        coord = self.filtered_coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.roi_transforms:
            img = self.roi_transforms(img)
            
        return {'img': img, 'coord': coord}


if __name__ == '__main__':
    # login with your User Access Token
    # found at https://huggingface.co/settings/tokens 
    login(token='<your_token>')  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path', 
        type=str, 
        required=True, 
        help='Path to CSV containing cases')
    parser.add_argument(
        '--h5_root', 
        type=str, 
        required=True, 
        help='Path to the location of h5 files (patches)')
    parser.add_argument(
        '--wsi_root', 
        type=str, 
        required=True, 
        help='Path to the location of WSIs in .svs format')
    parser.add_argument(
        '--conch_preds_root', 
        type=str, 
        required=True, 
        help='Path to the location of CONCH predictions in csv format.')
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help='Path to output predictions and/or features')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    args = parser.parse_args()

    # paths 
    h5_root = args.h5_root # generated tiles
    output_root = args.output_path 
    wsi_root = args.wsi_root 
    conch_preds_root = args.conch_preds_root

    # read WSIs to be processed 
    wsi_df = pd.read_csv(args.csv_path)

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.join(output_root, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(output_root, 'pt_files'))

    # get encoder and transform 
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True).to(device)
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    
    _ = model.eval()
    model = model.to(device)
    total = len(wsi_df)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    

    # iterate through WSIs 
    for i, row in wsi_df.iterrows(): 

        print(f"{i+1}/{len(wsi_df)}")
        print(f"slide id: {row['slide_id']}")

        pid = row['pid']
        slide_id = row['slide_id']
        h5_path = os.path.join(h5_root, str(pid), f'{str(slide_id)}.h5')
        wsi_path = os.path.join(wsi_root, str(pid), f'{str(slide_id)}.svs')
        conch_csv_path = os.path.join(conch_preds_root, str(pid), f'{str(slide_id)}.csv')
        bag_name = slide_id+'.h5'

        if not args.no_auto_skip and slide_id+'.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue  

        # specify output location 
        os.makedirs(os.path.join(output_root, 'h5_files', str(pid)), exist_ok=True)
        output_path = os.path.join(output_root, 'h5_files',  str(pid), bag_name)
        
        time_start = time.time()
        print('\ncomputing features for {}'.format(output_path))

        # initialize dataset 
        wsi = openslide.open_slide(wsi_path)
        dataset = Filtered_Whole_Slide_Bag_FP(file_path=h5_path, wsi=wsi, conch_preds_path=conch_csv_path, img_transforms=preprocess, print_summary=True)
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

        # iterate thru the tiles in this wsi 
        mode = 'w'
        print(f'processing a total of {len(dataloader)} batches'.format(len(dataloader)))
        logits_all, coords_all, preds_all = [], [], []
        for batch_idx, data in enumerate(tqdm(dataloader)): 
            with torch.inference_mode():	
                batch = data['img']
                coords = data['coord'].numpy().astype(np.int32)
                batch = batch.to(device, non_blocking=True)
                
                features = model(batch)
                features = features.cpu().numpy().astype(np.float32)

                asset_dict = {'features': features, 'coords': coords}
                save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
                mode = 'a'
        time_elapsed = time.time() - time_start
        print(f'...done! Took {time_elapsed} secconds')
        with h5py.File(output_path, "r") as file:
            features = file['features'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', file['coords'].shape)
        
        features = torch.from_numpy(features)
        bag_base, _ = os.path.splitext(bag_name)
        os.makedirs(os.path.join(output_root, 'pt_files', str(pid)), exist_ok=True)
        torch.save(features, os.path.join(output_root, 'pt_files', str(pid), bag_base+'.pt'))




