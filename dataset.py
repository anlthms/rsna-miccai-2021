import torch
from torch.utils import data as torch_data

from util import load_dicom_images_3d

class Dataset(torch_data.Dataset):
    def __init__(self, input_dir, paths, targets=None, mri_type=None, label_smoothing=0.01, split="train"):
        self.input_dir = input_dir
        self.paths = paths
        self.targets = targets
        self.mri_type = mri_type
        self.label_smoothing = label_smoothing
        self.split = split

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        scan_id = self.paths[index]
        if self.targets is None:
            data = load_dicom_images_3d(self.input_dir, str(scan_id).zfill(5), mri_type=self.mri_type[index], split=self.split)
        else:
            data = load_dicom_images_3d(self.input_dir, str(scan_id).zfill(5), mri_type=self.mri_type[index], split="train")

        if self.targets is None:
            return {"X": torch.tensor(data).float(), "id": scan_id}
        else:
            y = torch.tensor(abs(self.targets[index]-self.label_smoothing), dtype=torch.float)
            return {"X": torch.tensor(data).float(), "y": y}

