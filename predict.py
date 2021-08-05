import os
import multiprocessing as mp
import pandas as pd
import torch
from torch.utils import data as torch_data

from dataset import Dataset
from model import Model
from config import Config

# ## Predict function

def predict(input_dir, modelfile, df, mri_type, split, device, num_workers):
    if not os.path.exists(modelfile):
        # set path to match kaggle environment
        modelfile = f'../input/miccai/{modelfile}'
    assert os.path.exists(modelfile)
    print("Predict:", modelfile, mri_type, df.shape)
    df.loc[:,"MRI_Type"] = mri_type
    checkpoint = torch.load(modelfile)
    conf = Config(checkpoint["hp_dict"])

    data_retriever = Dataset(
        conf,
        input_dir,
        df.index.values,
        mri_type=df["MRI_Type"].values,
        split=split
    )

    data_loader = torch_data.DataLoader(
        data_retriever,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
    )

    model = Model(conf)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    y_pred = []
    ids = []

    for e, batch in enumerate(data_loader,1):
        #print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            tmp_pred = torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze()
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())
            ids.extend(batch["id"].numpy().tolist())

    preddf = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
    preddf = preddf.set_index("BraTS21ID")
    return preddf

def test(data_dir='../input', num_workers=4):
    kaggle_input_dir = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
    data_dir = kaggle_input_dir if os.path.exists(kaggle_input_dir) else data_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mri_types = ['FLAIR','T1w','T1wCE','T2w']
    modelfiles = [f'{mtype}-best.pth' for mtype in mri_types]

    # ## Ensemble for submission
    submission = pd.read_csv(f"{data_dir}/sample_submission.csv", index_col="BraTS21ID")

    submission["MGMT_value"] = 0
    for m, mtype in zip(modelfiles, mri_types):
        pred = predict(data_dir, m, submission, mtype, "test", device, num_workers)
        submission["MGMT_value"] += pred["MGMT_value"]

    submission["MGMT_value"] /= len(modelfiles)
    submission["MGMT_value"].to_csv("submission.csv")
    print('Saved submission.csv')

if __name__ == '__main__':
    test()