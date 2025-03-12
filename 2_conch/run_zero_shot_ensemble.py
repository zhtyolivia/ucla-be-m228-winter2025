import os
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import numpy as np 
import argparse
from PIL import Image
import h5py

from conch.open_clip_custom import create_model_from_pretrained
from conch.downstream.zeroshot_path import zero_shot_classifier, run_zeroshot
# from models import get_encoder

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import openslide 


CLASS_TO_IDX = {
    'non_tumor': 0, 
    'LUAD': 1, 
    'LUSC': 2, 
} 

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        file_path,
        wsi,
        img_transforms=None,
        print_summary=False):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            img_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path

        self.class_to_idx =  CLASS_TO_IDX
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
        if print_summary: 
           self.summary()
            
    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path,'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        img = self.roi_transforms(img)
        return {'img': img, 'coord': coord}
    
    def print_h5_contents(self, file):
        """Recursively print all datasets and attributes in the HDF5 file."""
        def recursive_print(name, obj):
            print(f"Dataset/Group: {name}")
            if isinstance(obj, h5py.Dataset):
                print(f"  - Shape: {obj.shape}")
                print(f"  - Dtype: {obj.dtype}")
            if obj.attrs:
                print(f"  - Attributes: {dict(obj.attrs)}")

        print("HDF5 File Structure:")
        file.visititems(recursive_print)


if __name__ == '__main__':

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
        '--output_path', 
        type=str, 
        required=True, 
        help='Path to output predictions')
    parser.add_argument(
        '--prompt_file', 
        type=str, 
        default='./prompts/nsclc_prompts_all_per_class.json', 
        help='Path to prompt file')
    parser.add_argument('--model_name', type=str, default='conch_v1', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_auto_skip', default=False, action='store_true')
    args = parser.parse_args()

    # paths 
    h5_root = args.h5_root 
    output_root = args.output_path 
    wsi_root = args.wsi_root 

    # get encoder and transform 
    hf_token = os.getenv("HF_AUTH_TOKEN") # your hugging face token 
    # alternatively, set the token directly 
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=hf_token, device=device)
    
    # load prompts 
    prompt_file = args.prompt_file 
    with open(prompt_file) as f:
        prompts = json.load(f)['0']

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    # read WSIs to be processed 
    wsi_df = pd.read_csv(args.csv_path)

    # iterate through WSIs 
    for i, row in wsi_df.iterrows(): 

        print(f"{i+1}/{len(wsi_df)}")
        print(f"slide id: {row['slide_id']}")

        pid = row['pid']
        slide_id = row['slide_id']
        h5_path = os.path.join(h5_root, str(pid), f'{str(slide_id)}.h5')
        wsi_path = os.path.join(wsi_root, str(pid), f'{str(slide_id)}.svs')

        # specify output location 
        output_dir_path = os.path.join(args.output_path, str(pid))
        os.makedirs(output_dir_path, exist_ok=True) 
        output_csv_path = os.path.join(output_dir_path, f'{slide_id}.csv')
        
        # initialize dataset 
        wsi = openslide.open_slide(wsi_path)
        dataset = Whole_Slide_Bag_FP(file_path=h5_path, wsi=wsi, img_transforms=preprocess, print_summary=(i==0))

        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        if hasattr(dataloader.dataset, 'class_to_idx'): 
            idx_to_class = {v:k for k,v in dataloader.dataset.class_to_idx.items()}
        else:
            raise ValueError('Dataset does not have label_map attribute')

        # classnames and prompts 
        classnames = prompts['classnames']
        templates = prompts['templates']
        n_classes = len(classnames)
        classnames_text = [classnames[str(idx_to_class[idx])] for idx in range(n_classes)]
        # print class names during the first iteration 
        if i == 0: 
            for class_idx, classname in enumerate(classnames_text):
                print(f'{class_idx}: {classname}')

        # load weights for zero-shot inference 
        zeroshot_weights = zero_shot_classifier(model, classnames_text, templates, device=device)
        # print(zeroshot_weights.shape)

        # iterate thru the tiles in this wsi 
        logits_all, coords_all, preds_all = [], [], []
        for batch_idx, data in enumerate(tqdm(dataloader)): 
            with torch.inference_mode():	
                batch = data['img']
                coords = data['coord'].numpy().astype(np.int32)
                batch = batch.to(device, non_blocking=True)
                
                image_features = model.encode_image(batch)
                logits = image_features @ zeroshot_weights
                preds = logits.argmax(dim=1)

                logits_all.append(logits.detach().cpu().numpy())
                preds_all.append(preds.detach().cpu().numpy())
                coords_all.append(coords)
        
        # save preds and targets 
        logits_all = np.concatenate(logits_all, axis=0)
        probs_all = F.softmax(torch.from_numpy(logits_all) * model.logit_scale.exp().item(), dim=1).numpy()
        preds_all = np.concatenate(preds_all, axis=0)
        coords_all = np.concatenate(coords_all, axis=0)

        df_predictions = pd.DataFrame({
            'x': [coord[0] for coord in coords_all],  # x-coord 
            'y': [coord[1] for coord in coords_all],  # y-coord 
            'pred_label': preds_all,  # predicted class index 
        })
        df_predictions.to_csv(output_csv_path, index=False)
        print(f"Saved model predictions to {output_csv_path}")

