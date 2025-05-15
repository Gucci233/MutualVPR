# MutualVPR

## 1. Environment Setup

Our experiments were conducted using **three NVIDIA RTX 3090 GPUs**. Please follow the steps below to configure the environment using the provided `requirements.txt` file:

```bash
conda create -n MutualVPR python=3.8
conda activate MutualVPR
pip install -r requirements.txt
```
---

## 2. Dataset Preparation

We use the **SF-XL** dataset in our experiments. It can be downloaded by following the instructions in the official [gmberton/CosPlace](https://github.com/gmberton/CosPlace) repository (CVPR 2022 paper: *"Rethinking Visual Geo-localization for Large-Scale Applications"*).

> ⚠️ **Note:** Panoramic images are included in the `raw` dataset folder.

### Optional: Multi-angle Cropping

To apply a multi-view cropping strategy, utility functions are provided in `datasets/dataset_utils.py`. You can manually set the following parameters:

- Starting angle for cropping  
- Saving path  
- Cropping step size, etc.

---

## 3. Training

Before training, make sure to modify the dataset paths in the code to match your local setup.


Run the training script with:
```bash
python train.py
--train_set_folder /your/path/train
--val_set_folder /your/path/val
--test_set_folder /your/path/test
```


---

## 4. Testing

The `datasets/` directory includes test splits for the following benchmarks:

- SF-XL  
- Tokyo 24/7  
- MSLS  
- Pitts250k  
- Pitts30k  

To test on any of these datasets, simply run:

```bash
python test.py
```