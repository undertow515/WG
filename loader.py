from torch.utils.data import Dataset, DataLoader
from typing import List
from configs.config_loader import Config
import torch
import pandas as pd
class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, config: Config, t: str = "train"):
        self.data = data
        self.data['Date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
        self.data = self.data.set_index('Date')
        self.src_len = config.src_len
        self.tgt_len = config.tgt_len
        # print(self.data.index)
        if t == "train":
            train_range = pd.date_range(start=config.train_start, end=config.train_end)
            self.data = self.data.loc[train_range]
        elif t == "val":
            val_range = pd.date_range(start=config.val_start, end=config.val_end)
            self.data = self.data.loc[val_range]
        elif t == "test":
            test_range = pd.date_range(start=config.test_start, end=config.test_end)
            self.data = self.data.loc[test_range]
        else:
            raise ValueError("Invalid data split")
        self.X = torch.tensor(self.data[config.input_variables].values, dtype=torch.float32)
        self.y = torch.tensor(self.data[config.target_variables].values, dtype=torch.float32)

        self.std_X = self.X.std(dim=0)
        self.mean_X = self.X.mean(dim=0)
        self.std_y = self.y.std(dim=0)
        self.mean_y = self.y.mean(dim=0)

        self.X = (self.X - self.mean_X) / self.std_X
        self.y = (self.y - self.mean_y) / self.std_y
        

    def __len__(self):
        return len(self.data) - self.src_len - self.tgt_len + 1

    def __getitem__(self, idx):
        src = self.X[idx:idx+self.src_len]
        tgt = self.y[idx+self.src_len:idx+self.src_len+self.tgt_len]
        return src, tgt