import wandb
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

from src.utils import setup_logger
from src.models.utils import (
    get_device,
    compile_model,
    data_parallel,
    save_model,
)
from src.optimizers import ADOPT

logger = setup_logger(__name__)


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Dataset,
        config: OmegaConf,
        wandb_run: tuple[str, wandb.sdk.wandb_run.Run] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        # Includes saving the model and evaluation
        self.checkpoint_every_n_samples = self.config.training.checkpoint_every_n_samples

        # Training results and metrics
        self.train_metrics = {}
        self.test_metrics = {}
        self.evaluation_callbacks = []

        # Dataset setup
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=True,
            pin_memory=True
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            shuffle=False,
            pin_memory=True
        )
        logger.info(f"Train samples: {len(self.train_dataset)} Test samples: {len(self.test_dataset)}")
        logger.info(f"Train steps: {len(self.train_dataloader)} Test steps: {len(self.test_dataloader)}")

        # Model setup
        self.device = get_device(
            device_id=self.config.device,
            force_cpu=self.config.force_cpu
        )
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)
        # self.model = compile_model(self.model, self.config)
        self.model = data_parallel(self.model, self.device, self.config)

        # Training setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=1e-4
        )
        
        self.scheduler = None
        if getattr(self.config.training, "lr_scheduler", True):
            # Calculate one cycle length - ideally 0.5 cycles per epoch
            steps_size_up = len(self.train_dataloader)
            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.config.training.lr / 10,
                max_lr=self.config.training.lr,
                step_size_up=steps_size_up, # steps size down is the same
            )
            
        # self.register_hooks()
        
        # Wandb setup
        # Existent run - for evaluation
        if wandb_run is not None:
            self.run_id, self.wandb = wandb_run
            return None

        # Start new run
        self.run_id = wandb.util.generate_id()
        self.run_name = (
            f"{self.config.name}_"
            f"{self.config.dataset.name}_"
            f"{self.config.model.name}_"
            f"{self.config.experiment_name}_"
            f"{self.run_id}"
        )
        self.wandb = wandb.init(
            entity="",
            project="",
            name=self.run_name,
            config=OmegaConf.to_container(self.config, resolve=True),
            mode="disabled" if self.config.debug else "run",
        )

    def register_hooks(self):
        # Gradient logging
        if self.config.debug:
            def log_grad(grad, param, name):
                logger.info(
                    f"{name} GRAD -> mean: {grad.mean().item():.6f}, "
                    f"min: {grad.min().item():.6f}, max: {grad.max().item():.6f}"
                )
                logger.info(f"{name} GRAD shape: {grad.shape}")
                logger.info(
                    f"{name} GRAD values: "
                    f"{grad.view(-1)[:8].round(decimals=5).cpu()}"
                )
                logger.info(
                    f"{name} PARAM values: "
                    f"{param.view(-1)[:8].round(decimals=5).cpu()}"
                )

            for name, param in self.model.named_parameters():
                param.register_hook(lambda grad, p=param, n=name: log_grad(grad, p, n))

    def optimizer_step(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
            
    def add_evaluation_callbacks(self, callbacks: list[dict]) -> None:
        for callback in callbacks:
            self.evaluation_callbacks.append(callback)

    def save_checkpoint(self, checkpoint_name: str = None) -> None:
        """
        Saves the model and config to disk.
        Args:
            checkpoint_name: Name of the saved checkpoint folder.
        """
        save_model(self.model, self.config, checkpoint_name)        

    def run(self) -> None:
        """ Main training loop """
        raise NotImplementedError
    
    def train(self) -> None:
        """ Training epoch """
        raise NotImplementedError
    
    def test(self) -> None:
        """ Testing epoch """
        raise NotImplementedError

    def evaluate(self) -> None:
        """ Additional evaluation """
        raise NotImplementedError
