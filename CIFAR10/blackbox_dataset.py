import torch
from torch.utils.data import Dataset
import pickle

class BlackboxDataset(Dataset):
    def __init__(self, filepath, transform=None):
        super(BlackboxDataset, self).__init__()
        self.transform = transform
        with open(filepath, 'rb') as f:
            saved_adv = pickle.load(f)
        self.adv_data, self.label = saved_adv
        assert len(self.adv_data) == len(self.label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.adv_data[idx]
        label = self.label[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, label
