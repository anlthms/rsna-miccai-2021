#!/usr/bin/env python
# coding: utf-8

# ## Use stacked images (3D) and Efficientnet3D model
#
# Acknowledgements:
#
# - https://www.kaggle.com/ihelon/brain-tumor-eda-with-animations-and-modeling
# - https://www.kaggle.com/furcifer/torch-efficientnet3d-for-mri-no-train
# - https://github.com/shijianjian/EfficientNet-PyTorch-3D
#
#
# Use models with only one MRI type, then ensemble the 4 models
#

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp

import torch
from torch.utils import data as torch_data
from torch.nn import functional as torch_functional

from sklearn import model_selection as sk_model_selection
from sklearn.metrics import roc_auc_score

from model import Model
from dataset import Dataset
from util import set_seed, load_dicom_image, load_dicom_images_3d
from predict import predict

parser = argparse.ArgumentParser()
parser.add_argument(
    '-b', '--batch-size', default=4, type=int, help='mini-batch size')
parser.add_argument(
    '-j', '--num_workers', default=mp.cpu_count(), type=int, metavar='N',
    help='number of data loading workers')
parser.add_argument(
    '--seed', default=None, type=int,
    help='seed for initializing the random number generator')
parser.add_argument(
    '--epochs', default=10, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '--input', default='../input', metavar='DIR', help='input directory')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(
        self,
        model,
        device,
        optimizer,
        criterion
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None

    def fit(self, epochs, train_loader, valid_loader, save_path, patience):
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s            ",
                n_epoch, train_loss, train_time
            )

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

            # if True:
            # if self.best_valid_score < valid_auc:
            if self.best_valid_score > valid_loss:
                self.save_model(n_epoch, save_path, valid_loss, valid_auc)
                self.info_message(
                     "auc improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_score, valid_loss, self.lastmodel
                )
                self.best_valid_score = valid_loss
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message("\nValid auc didn't improve last {} epochs.", patience)
                break

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device)
            targets = batch["y"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            sum_loss += loss.detach().item()

            self.optimizer.step()

            #message = 'Train Step {}/{}, train_loss: {:.4f}'
            #self.info_message(message, step, len(train_loader), sum_loss/step, end="\r")

        return sum_loss/len(train_loader), int(time.time() - t)

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device)
                targets = batch["y"].to(self.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                sum_loss += loss.detach().item()
                y_all.extend(batch["y"].tolist())
                outputs_all.extend(outputs.tolist())

            #message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            #self.info_message(message, step, len(valid_loader), sum_loss/step)

        y_all = [1 if x > 0.5 else 0 for x in y_all]
        auc = roc_auc_score(y_all, outputs_all)

        return sum_loss/len(valid_loader), auc, int(time.time() - t)

    def save_model(self, n_epoch, save_path, loss, auc):
        self.lastmodel = f"{save_path}-best.pth"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
                "auc": auc,
                "loss": loss,
            },
            self.lastmodel,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


def train_mri_type(df_train, df_valid, mri_type, args):
    if mri_type=="all":
        train_list = []
        valid_list = []
        for mri_type in mri_types:
            df_train["MRI_Type"] = mri_type
            train_list.append(df_train.copy())
            df_valid["MRI_Type"] = mri_type
            valid_list.append(df_valid.copy())

        df_train = pd.concat(train_list)
        df_valid = pd.concat(valid_list)
    else:
        df_train["MRI_Type"] = mri_type
        df_valid["MRI_Type"] = mri_type

    print(df_train.shape, df_valid.shape)
    print(df_train.head())

    train_data_retriever = Dataset(
        args.input,
        df_train["BraTS21ID"].values,
        df_train["MGMT_value"].values,
        df_train["MRI_Type"].values
    )

    valid_data_retriever = Dataset(
        args.input,
        df_valid["BraTS21ID"].values,
        df_valid["MGMT_value"].values,
        df_valid["MRI_Type"].values
    )

    train_loader = torch_data.DataLoader(
        train_data_retriever,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    valid_loader = torch_data.DataLoader(
        valid_data_retriever,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = Model()
    model.to(device)

    #checkpoint = torch.load("best-model-all-auc0.555.pth")
    #model.load_state_dict(checkpoint["model_state_dict"])

    #print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch_functional.binary_cross_entropy_with_logits

    trainer = Trainer(
        model,
        device,
        optimizer,
        criterion
    )

    history = trainer.fit(
        args.epochs,
        train_loader,
        valid_loader,
        f"{mri_type}",
        10,
    )

    return trainer.lastmodel

def main():
    args = parser.parse_args()

    mri_types = ['FLAIR','T1w','T1wCE','T2w']
    print(load_dicom_images_3d(args.input, "00000").shape)

    if args.seed:
        set_seed(args.seed)

    # ## train / test splits

    train_df = pd.read_csv(f"{args.input}/train_labels.csv")

    df_train, df_valid = sk_model_selection.train_test_split(
        train_df,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_df["MGMT_value"],
    )

    df_train = df_train.copy()
    df_valid = df_valid.copy()
    df_train.tail()

    modelfiles = [train_mri_type(df_train, df_valid, m, args) for m in mri_types]
    print(modelfiles)

    # ## Ensemble for validation

    df_valid = df_valid.set_index("BraTS21ID")
    df_valid["MGMT_pred"] = 0
    for m, mtype in zip(modelfiles,  mri_types):
        pred = predict(args.input, m, df_valid, mtype, "train", device, args.num_workers)
        df_valid["MGMT_pred"] += pred["MGMT_value"]
    df_valid["MGMT_pred"] /= len(modelfiles)
    auc = roc_auc_score(df_valid["MGMT_value"], df_valid["MGMT_pred"])
    print(f"Validation ensemble AUC: {auc:.4f}")
    sns.displot(df_valid["MGMT_pred"])


if __name__ == '__main__':
    main()
    print('Done')
