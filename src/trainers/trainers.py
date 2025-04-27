import gc
import os

from omegaconf import OmegaConf
import numpy as np
import wandb
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt

from src.metrics import (
    MetricTracker,
    accuracy,
    cosine_similarity,
    cosine_similarity_loss,
    MSECosineLoss,
)
from src.paths import PATHS
from src.trainers.base import BaseTrainer
from src.utils import setup_logger, log_table

logger = setup_logger(__name__)


class ClassifierTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        config: OmegaConf
    ):
        super().__init__(model, train_dataset, test_dataset, config)

    def run(self) -> None:
        """ Main training loop """
        self.criterion = nn.CrossEntropyLoss()

        self.metric_tracker = MetricTracker()
        self.metric_tracker.register_metric("cross_entropy_loss", nn.CrossEntropyLoss())
        self.metric_tracker.register_metric("accuracy", accuracy)

        # training loop
        for epoch in range(self.config.training.epochs):
            self.train(epoch)
            self.test(epoch)
            self.evaluate(epoch)

        # Save the model
        if self.config.training.save_final_model:
            self.save_checkpoint(f"{self.run_id}_final")

    def train(self, epoch: int) -> None:
        """ Training epoch """
        self.metric_tracker.reset()
        self.model.train()

        accumulation_steps = getattr(
            self.config.training, "gradient_accumulation_steps", 1
        )

        for step, (inputs, captions, id, labels) in enumerate(tqdm(self.train_dataloader)):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            labels = labels.long()

            if step % accumulation_steps == 0:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or \
                (step + 1) == len(self.train_dataloader):
                self.optimizer.step()

            batch_metrics = self.metric_tracker.update(outputs, labels)
            batch_metrics = {
                f"{k}_step_train": round(v, 4) 
                for k, v in batch_metrics.items()
            }
            self.wandb.log(batch_metrics)
            if self.config.debug:
                logger.info(f"Step: {step} - {batch_metrics}")

        # Logging aggregated metrics
        logger.info(f"Epoch: {epoch+1}/{self.config.training.epochs} - Training")
        epoch_metrics = self.metric_tracker.aggregate()
        epoch_metrics = {f"{k}_train": round(v, 4) for k, v in epoch_metrics.items()}
        epoch_metrics["epoch"] = epoch
        self.wandb.log(epoch_metrics)
        log_table(epoch_metrics, logger)
        
        # Save epoch metrics
        self.train_metrics[epoch] = epoch_metrics

    def test(self, epoch: int) -> None:
        """ Testing epoch """
        self.metric_tracker.reset()
        self.model.eval()

        labels_list = []
        outputs_list = []
        with torch.no_grad():
            for step, (inputs, captions, id, labels) in enumerate(tqdm(self.test_dataloader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long()

                outputs = self.model(inputs)
                
                labels_list.extend(labels.cpu().numpy())
                outputs_list.extend(outputs.argmax(dim=1).cpu().numpy())
                
                batch_metrics = self.metric_tracker.update(outputs, labels)
                batch_metrics = {
                    f"{k}_step_test": round(v, 4)
                    for k, v in batch_metrics.items()
                }
                if self.config.debug:
                    logger.info(f"Step: {step} - {batch_metrics}")

        # Logging aggregated metrics
        logger.info(f"Epoch: {epoch+1}/{self.config.training.epochs} - Testing")
        epoch_metrics = self.metric_tracker.aggregate()
        epoch_metrics = {f"{k}_test": round(v, 4) for k, v in epoch_metrics.items()}
        epoch_metrics["epoch"] = epoch
        self.wandb.log(epoch_metrics)
        log_table(epoch_metrics, logger)
        
        # # Confusion matrix
        # cm = confusion_matrix(labels_list, outputs_list)
        # plt.figure(figsize=(10,10))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.title(f'Confusion Matrix - Epoch {epoch}')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')

        # os.makedirs(PATHS.checkpoints, exist_ok=True)
        # save_path = os.path.join(PATHS.checkpoints, f"{self.run_id}_{epoch:03}_cm.png")
        # plt.savefig(save_path)
        # plt.close()

        # self.wandb.log({"confusion_matrix": wandb.Image(save_path)})
        
        # Save epoch metrics
        self.test_metrics[epoch] = epoch_metrics

    def evaluate(self, epoch: int) -> None:
        """ Evaluate the model on the evaluation callbacks """
        if self.evaluation_callbacks:
            logger.info(f"Epoch: {epoch+1}/{self.config.training.epochs} - Evaluating")
            for callback in self.evaluation_callbacks:
                callback["func"](*callback["args"])


class ZeroShotEvaluation(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        config: OmegaConf,
        wandb_run: tuple[str, wandb.sdk.wandb_run.Run] = None,
    ):
        super().__init__(model, train_dataset, test_dataset, config, wandb_run)
        if not hasattr(self.model, "encode"):
            raise ValueError("Model must have an `encode` method implemented.")

    @staticmethod
    def top_k_accuracy(
        true_labels: torch.Tensor, pred_labels: torch.Tensor, k: int
    ) -> float:
        """ Calculate top-k accuracy """
        correct = 0
        for i in range(len(true_labels)):
            if true_labels[i] in pred_labels[i][:k]:
                correct += 1
        return correct / len(true_labels)
    
    def calculate_embeddings(
        self, dataset: Dataset
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Calculate embeddings for the dataset """
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True
        )

        embeddings = []
        labels = []
        with torch.no_grad():
            for step, (inputs, caption, id, targets) in enumerate(tqdm(dataloader)):
                inputs = inputs.to(self.device)
                outputs = self.model.encode(inputs)

                # Eqaualize dimensions for the knn
                if outputs.dim() == 2:
                    outputs = outputs.unsqueeze(1)

                # Patch reduction
                outputs, _ = outputs.max(dim=1)

                embeddings.append(outputs.detach().cpu())
                labels.append(targets)

        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        return embeddings, labels

    def run(self) -> None:
        """ Main evaluation loop """
        
        train_embeddings, train_labels = self.calculate_embeddings(self.train_dataset)
        test_embeddings, test_labels = self.calculate_embeddings(self.test_dataset)
        
        # Nearest Neighbors
        knn = NearestNeighbors(n_neighbors=5, metric="cosine")
        knn.fit(train_embeddings)
        
        # Find the top-k neighbors for each test sample
        distances, indices = knn.kneighbors(test_embeddings)

        # Predict the labels of the nearest neighbors
        pred_labels = train_labels[indices]

        # Calculate top-1, top-3, and top-5 accuracy
        acc1 = ZeroShotEvaluation.top_k_accuracy(test_labels, pred_labels, k=1)
        acc3 = ZeroShotEvaluation.top_k_accuracy(test_labels, pred_labels, k=3)
        acc5 = ZeroShotEvaluation.top_k_accuracy(test_labels, pred_labels, k=5)
        logger.info(f"Top-1 Accuracy: {acc1 * 100:.2f}%")
        logger.info(f"Top-3 Accuracy: {acc3 * 100:.2f}%")
        logger.info(f"Top-5 Accuracy: {acc5 * 100:.2f}%")
        
        self.wandb.log({
            f"top1_accuracy_{self.config.dataset.name}": acc1,
            f"top3_accuracy_{self.config.dataset.name}": acc3,
            f"top5_accuracy_{self.config.dataset.name}": acc5
        })
