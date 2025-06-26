import torch
import os
from tqdm import tqdm
import torch.nn as nn
from src.model.dataloader import DataLoaders
from src.utils.utils import get_device


class Trainer:

    def __init__(self, model ,pos, evals, num_epochs=10, batch_size=64):
        self.num_epochs = num_epochs
        self.device = get_device()
        self.model = model.to(self.device)

        dl = DataLoaders(pos, evals, batch_size=batch_size, shuffle=True)

        self.loss_fn = nn.HuberLoss(delta=1.0) # mixing MSE and MAE to strike balance between mate and non mate
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_loader = dl.train_dataloader()
        self.test_loader = dl.test_dataloader()
        self.val_loader = dl.val_dataloader()

    def train(self, store_epoch=False):
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for pos_state, pos_evals in progress_bar:

                pos_state, pos_evals = pos_state.to(self.device), pos_evals.to(self.device)
                self.optimizer.zero_grad()
                out_evals = self.model(pos_state)
                loss = self.loss_fn(out_evals, pos_evals)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            self.train_losses.append(epoch_loss / len(self.train_loader))

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_state, val_evals in self.val_loader:
                    val_state, val_evals = val_state.to(self.device), val_evals.to(self.device)
                    val_outputs = self.model(val_state)
                    loss = self.loss_fn(val_outputs, val_evals)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(self.val_loader)
            self.val_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss:.6f}")

            if store_epoch:
                torch.save(self.model.state_dict(), f"{os.getenv("TRAIN_EPOCH_PATH")}/model_epoch_{epoch + 1}.pth")







