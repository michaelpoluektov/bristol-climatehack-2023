# ClimateHack submission

## Disclaimer

This is far from polished. We would not recommend for anyone to reproduce this as-is, but there are a few tricks that might be useful to other people building similar models.

Assuming you are familiar with Perceiver/GQA/Transformer-based models, you probably won't find most of this repository interesting. The 2 tricks that are somewhat original are as follows:

- Fancy activation function in `model/model.py`, defined in the `FancyActivation` class.
- Image stacking for faster training, defined in `model/conv.py`.

## Directory structure

```
.
├── README.md               # This file
├── dataset.py              # Handles data loading and preprocessing
├── del_bad.sh              # Hacky script to delete days with missing data
├── downloader.py           # Download datasets
├── model
│   ├── __init__.py
│   ├── attention.py        # Implements GQA
│   ├── blocks.py           # GQA blocks
│   ├── conv.py             # Conv blocks for image frontends
│   ├── weight_init.py      # Weight initialisation helper function
│   ├── model.py            # Model architecture
│   └── model_lightning.py  # PyTorch Lightning wrapper
├── poetry.lock             # Dependency lock for Poetry
├── prep.py                 # Preprocessing script for dataset
├── pyproject.toml          # Project configuration for Poetry
├── slides.pdf              # Presentation slides
├── train.py                # Training script
└── utils.py                # Utility functions for preprocessing (also used in evaluation)
```

## Reproducing leaderboard results

### Setup environment

Download and install `poetry`, then run `poetry install` and poetry `shell` inside the directory.

### Download dataset

When given the option to select which datasets, PV and HRV must be selected. No other datasets were used.

```bash
mkdir -p dataset
# download local validation
wget -O dataset/local_validation.hdf5 https://github.com/climatehackai/getting-started-2023/releases/download/v1.0.0/data.hdf5

# download indices
wget https://raw.githubusercontent.com/climatehackai/getting-started-2023/main/indices.json -O dataset/indices.json
# download datasets from hugging face, select PV and HRV
python downloader.py
```

### Prep dataset

The next step is to preprocess the data ready for training.

```bash
python prep.py
```

### Training

```bash
python train.py
```

