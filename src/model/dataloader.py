import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

class DataLoaders():
    def __init__(self, X, y, batch_size=64, shuffle=True, test_size=0.2, **loader_kwargs):

        # First split: train (80%) and temp (20%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)

        # Second split: validation (10%) and test (10%) from temp
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, **loader_kwargs)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


