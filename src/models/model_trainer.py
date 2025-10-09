import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

class PlantModelTrainer:
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['model']['learning_rate'],
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        self.train_losses = []
        self.val_losses = []
        self.metrics_history = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        return running_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calcular métricas
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return val_loss / len(dataloader), metrics

    def train(self, train_loader, val_loader, epochs):
        print("Iniciando treinamento...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Atualizar histórico de métricas
            for key in self.metrics_history:
                self.metrics_history[key].append(metrics[key])

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            print(f'  Accuracy: {metrics["accuracy"]:.4f}')
            print(f'  F1-Score: {metrics["f1"]:.4f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')

    def plot_training_history(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Plot metrics
        epochs_range = range(1, len(self.metrics_history['accuracy']) + 1)
        ax2.plot(epochs_range, self.metrics_history['accuracy'], label='Accuracy')
        ax2.set_title('Accuracy')

        ax3.plot(epochs_range, self.metrics_history['precision'], label='Precision', color='orange')
        ax3.plot(epochs_range, self.metrics_history['recall'], label='Recall', color='green')
        ax3.set_title('Precision and Recall')
        ax3.legend()

        ax4.plot(epochs_range, self.metrics_history['f1'], label='F1-Score', color='red')
        ax4.set_title('F1-Score')

        plt.tight_layout()
        plt.show()