# MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering

## Environment Setup

Our experiments were conducted using **three NVIDIA RTX 3090 GPUs**. Please follow the steps below to configure the environment using the provided `requirements.txt` file:

```bash
conda create -n MutualVPR python=3.8
conda activate MutualVPR
pip install -r requirements.txt
```
---

## Dataset Preparation

We use the **SF-XL** dataset in our experiments. It can be downloaded by following the instructions in the official [gmberton/CosPlace](https://github.com/gmberton/CosPlace) repository (CVPR 2022 paper: *"Rethinking Visual Geo-localization for Large-Scale Applications"*).


The base dataset used in the experiments is derived from `processed` dataset folder, and the cropping strategy requires cropping the panoramic images.
Panoramic images are included in the `raw` dataset folder.

### Optional: Multi-angle Cropping

To apply a multi-view cropping strategy, utility functions are provided in `datasets/dataset_utils.py`. You can manually set the following parameters:

- Starting angle for cropping  
- Saving path  
- Cropping step size, etc.

After preparing the data, modify the initialize() function in train_dataset.py to include the newly cropped data.



## Training

Before training, make sure to modify the dataset paths in the code to match your local setup.


Run the training script with:
```bash
python train.py
--train_set_folder /your/path/train
--val_set_folder /your/path/val
--test_set_folder /your/path/test
```



## 🔗 Model Weights

Our pretrained weight file can be downloaded from the following link:

👉 [Download from Google Drive](https://drive.google.com/file/d/1fn67GO6sA3qIIIgOuM9VjmTHoEkkyopF/view?usp=drive_link)



## Testing

To evaluate SF-XL-testv1 and SF-XL-occlusion, you also need to download `queries-v1` and `queries-occlusion` from [CosPlace](https://github.com/gmberton/CosPlace). Both of them use the same database.

The `datasets/` directory includes test splits for the following benchmarks:

- SF-XL  
- Tokyo 24/7  
- MSLS  
- Pitts250k  
- Pitts30k  

To test on any of these datasets, select your checkpoints and simply run:

```bash
python test.py
```

## Acknowledgement

We would like to thank the authors of [CosPlace](https://github.com/gmberton/CosPlace) and [EigenPlaces](https://github.com/gmberton/EigenPlaces) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation.