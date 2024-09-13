import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import mlflow
from torchmetrics.classification import BinaryPrecision, BinaryRecall


class Trainer:
    def __init__(self, model, device, train_dataset, val_dataset, batch_size=32, learning_rate=1e-3, patience=100):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=10)
        self.patience = patience
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False
        self.best_model_wts = None

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_step(epoch)
            val_loss = self.validate_step(epoch)

            # Early stopping & saving best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.best_model_wts = self.model.state_dict()
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                self.early_stop = True
                break

        # Load best weights
        if self.best_model_wts is not None:
            self.model.load_state_dict(self.best_model_wts)
            torch.save(self.best_model_wts, 'best_model.pth')

    def train_step(self, epoch):
        self.model.train()
        total_loss = 0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(1), target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        mlflow.log_metric('epoch_train_loss', avg_loss, step=epoch + 1)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    def validate_step(self, epoch):
        self.model.eval()
        total_loss = 0
        precision = 0
        recall = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(1), target)
                total_loss += loss.item()

                pred = torch.sigmoid(output.squeeze(1))
                precision += BinaryPrecision(threshold=0.5)(pred.cpu(), target.cpu()).item()
                recall += BinaryRecall(threshold=0.5)(pred.cpu(), target.cpu()).item()

        avg_loss = total_loss / len(self.val_loader)
        avg_precision = precision / len(self.val_loader)
        avg_recall = recall / len(self.val_loader)

        mlflow.log_metric('epoch_val_loss', avg_loss, step=epoch + 1)
        mlflow.log_metric('precision', avg_precision, step=epoch + 1)
        mlflow.log_metric('recall', avg_recall, step=epoch + 1)

        print(f'Validation loss: {avg_loss}')
        return avg_loss
