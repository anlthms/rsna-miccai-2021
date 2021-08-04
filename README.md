# Brain tumor classification

This repo contains code to train models for the [RSNA-MICCAI challenge](https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification).

# Acknowledgement

The code is adapted almost verbatim from a [notebook](https://www.kaggle.com/rluethy/efficientnet3d-with-one-mri-type) by [Roland Luethy](https://www.kaggle.com/rluethy).

# Usage

The following directory structure is assumed:
```
    code
        train.py
        train.sh
        ...
    input
        test
        train
        ...
```
- Download dataset and extract into `input`
- Run `install.sh` to install prerequisites
- Run `train.sh` to train a model
- Replace `<kaggle_userid>` inside `code/dataset-metadata.json` with your own Kaggle userid
- Run `kaggle_init.sh` to create a new dataset on Kaggle
- Run `kaggle_upload.sh` to upload code and trained models to Kaggle
- Use `kaggle_submit.ipynb` on the Kaggle platform to make a submission