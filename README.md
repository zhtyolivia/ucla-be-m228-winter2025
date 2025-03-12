# ucla-be-m228-winter2025

## Step 1: Setting up environment 

Create a docker container using [this docker image](https://hub.docker.com/layers/pytorch/pytorch/2.0.1-cuda11.7-cudnn8-devel/images/sha256-4f66166dd757752a6a6a9284686b4078e92337cd9d12d2e14d2d46274dfa9048). You can use the following command: 

```
docker run --shm-size=50g --gpus all -it --rm -v /:/workspace -v /etc/localtime:/etc/localtime:ro pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime 
```

Then, install dependencies: 
```
chmod +x install_pkgs.sh 
./install_pkgs.sh
```

## Step 2: Data preperation 

The TCGA-LUAD, TCGA-LUSC, CPTAC-LUAD, CPTAC-LSCC datasets can be downloaded from the [GDC Data Portal](https://portal.gdc.cancer.gov/). After you download the data, please organize into the following structure for data preprocessing: 
```
ucla-be-m228-winter2025/
│-- wsi/
│   │-- tcga/
│   │   │-- luad/
│   │   │   │-- <pid>/
│   │   │   │   │-- <slide_id>.svs
│   │   │   │   │-- <slide_id>.svs
│   │   │   │-- ...
│   │   │-- lusc/
│   │   │   │-- <pid>/
│   │   │   │   │-- <slide_id>.svs
│   │   │   │-- ...
│   │-- cptac/
│   │   │-- luad/
│   │   │   │-- ...
│   │   │-- lusc/
│   │   │   │-- ...
│-- 1_preprocessing/
│   │-- ...
```

## Step 3: Generate tiles  

In order to tile the WSIs, we used the [CLAM](https://github.com/mahmoodlab/CLAM) toolbox. First, clone the CLAM repository into `1_preprocessing` using: 
```
git clone git@github.com:mahmoodlab/CLAM.git
```

Next, in `./1_preprocessing`, run tiling by
```
python extract_tiles.py \
--patch \
--seg \
--stitch \
--no_auto_skip \
--preset ./CLAM/presets/tcga.csv \
--save_dir ../data \
```
Then the file structure looks like: 
```
ucla-be-m228-winter2025/
│-- wsi/
│   │-- ...
│-- generated_tiles/
│   │-- ...
│-- 1_preprocessing/
│   │-- ...
```

## Step 4: Run CONCH 
First, clone CONCH into `2_conch`. Then, in `2_conch`, run the following command: 
```
CUDA_VISIBLE_DEVICES=0 python run_zero_shot_ensemble.py \
--h5_root '../generated_tiles/tcga/luad/patches' \
--wsi_root '../data/TCGA/pathology/LUAD/' \
--output_path '../conch_preds/tcga/luad' \
--model_name 'conch_v1' --batch_size 256 --no_auto_skip
```

## Step 5: Feature extraction 
TO extract features from the tumor tiles (according to the predictions by CONCH), in `3_feature_extraction`, run the following: 
```
CUDA_VISIBLE_DEVICES=0 python main_uni_filter.py \
--h5_root '../generated_tiles/tcga/luad/patches' \
--wsi_root '../wsi/TCGA/pathology/LUAD/' \
--conch_preds_root '../conch_preds/tcga/luad' \
--output_path '..data/tcga/luad' \
--batch_size 256 --no_auto_skip
```

## Modeling 
To run our logistic regression and MLP classification models, navigate to `4_modeling` and run 
```
CUDA_VISIBLE_DEVICES=0 python mlp.py \
--tcga_root ../data/tcga \
--cptac_root ../data/cptac \
--exp ../exp/<experiment_name> \
--num_epochs 2 
```
See respective python script for specifications on other arguments. 
