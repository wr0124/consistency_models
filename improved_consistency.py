import hashlib
import logging

logging.basicConfig(level=logging.INFO)

import json
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Tuple, Union
import torch

torch.set_float32_matmul_precision("medium")
from einops import rearrange
from einops.layers.torch import Rearrange
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from UNet import UNet, UNetConfig
from consistency_models import (
    ConsistencySamplingAndEditing,
    ImprovedConsistencyTraining,
    pseudo_huber_loss,
)
from consistency_models.utils import update_ema_model_

from visdom import Visdom
import numpy as np
import torchvision.utils as vutils

from visdom import Visdom
from options import parse_opts
args = parse_opts()
viz = Visdom(env=args.env)
viz.line([0.0], [0.0], win="consis_loss", opts=dict(title="loss over time"))

from options import parse_opts

# dataModule
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True


class ImageDataModule(LightningDataModule):
    def __init__(self, config: ImageDataModuleConfig) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage: str = None) -> None:
        transform = T.Compose(
            [
                T.Resize(self.config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: (x * 2) - 1),
            ]
        )
        self.dataset = ImageFolder(self.config.data_dir, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )
# LitUNet
@dataclass
class LitImprovedConsistencyModelConfig:
    lr: float = args.lr
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = args.lr_scheduler_start_factor
    lr_scheduler_iters: int = args.lr_scheduler_iters
    sample_every_n_steps: int = args.sample_every_n_steps
    num_samples: int = args.num_samples
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )


class LitImprovedConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ImprovedConsistencyTraining,
        consistency_sampling: ConsistencySamplingAndEditing,
        unet_config: UNetConfig,
        config: LitImprovedConsistencyModelConfig,
        dataset_name: str,  # Add this parameter
        checkpoint_dir: str,  # And this one
    ) -> None:
        super().__init__()

        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.model = UNet(unet_config, checkpoint_dir)
        self.ema_model = UNet(unet_config, checkpoint_dir)
        self.config = config
        self.dataset_name = dataset_name  # Set as an instance attribute
        self.checkpoint_dir = checkpoint_dir  # Set as an instance attribute

        # Freeze the EMA model and set it to eval mode
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model = self.ema_model.eval()

    def training_step(self, batch: Union[Tensor, List[Tensor]], batch_idx: int) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        output = self.consistency_training(
            self.model, batch, self.global_step, self.trainer.max_steps
        )

        loss = (
            pseudo_huber_loss(output.predicted, output.target) * output.loss_weights
        ).mean()

        self.log_dict({"train_loss": loss, "num_timesteps": output.num_timesteps})
        # print("type of loss {} and time {}".format( loss.item(), self.global_step))
        viz.line([loss.item()], [self.global_step], win="consis_loss", update="append")
        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)
        # self.ema_model=update_ema_model_(self.ema_model, self.model, self.config.ema_decay_rate)

        if (
            (self.global_step + 1) % self.config.sample_every_n_steps == 0
        ) or self.global_step == 0:
            self.__sample_and_log_samples(batch)
            self.save_model_checkpoint()

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.config.lr_scheduler_start_factor,
            total_iters=self.config.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step", "frequency": 1}

        return [opt], [sched]

    @torch.no_grad()
    def __sample_and_log_samples(self, batch: Union[Tensor, List[Tensor]]) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.config.num_samples, batch.shape[0])

        # Log ground truth samples
        self.__log_images(
            batch[:num_samples].detach().clone(),
            "ground_truth",
            self.global_step,
            "ground_truth_window",
            window_size=(200, 200),
        )
        # option1 for sampling
        # self.ema_model.load_state_dict(self.model.state_dict())
        for sigmas in self.config.sampling_sigmas:
            
            samples = self.consistency_sampling(
                self.model,
                batch,
                sigmas,
                clip_denoised=True,
                verbose=True,
            )  # Generated samples
            samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
            self.__log_images(
                samples,
                f"generated_samples-sigmas={sigmas}",
                self.global_step,
                f"generated_samples_window_{sigmas}",
                window_size=(400, 400),
            )

    @torch.no_grad()
    def __log_images(
        self,
        images: Tensor,
        title: str,
        global_step: int,
        window_name: str,
        window_size: Tuple[int, int],
    ) -> None:
        images = images.detach().float()

        grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True, nrow=8
        )

        grid = grid.cpu().numpy()
        if grid.min() < 0:
            grid = (grid + 1) / 2
        width, height = window_size
        viz.images(
            grid,
            nrow=4,
            opts=dict(
                title=f"{title} at step {global_step}",
                caption=f"{title}",
                width=width,
                height=height,
            ),
            win=window_name,
        )
        # self.logger.experiment.add_image(title, grid, global_step)

    @torch.no_grad()
    def save_model_checkpoint(self):
        # Create a folder with the dataset name in the checkpoints directory
        checkpoint_folder = os.path.join(self.checkpoint_dir, self.dataset_name)
        os.makedirs(checkpoint_folder, exist_ok=True)
        # Save the checkpoint with the global step in the name
        checkpoint_filename = f"model_step_{self.global_step}.ckpt"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_filename)
        self.trainer.save_checkpoint(checkpoint_path)

        # Save the model using only the checkpoint filename
        # self.model.save_pretrained(checkpoint_filename)

        logging.info(f"Checkpoint saved: {checkpoint_path}")
        # Update the latest checkpoint symlink
        latest_path = os.path.join(checkpoint_folder, "latest.ckpt")
        if os.path.islink(latest_path):
            os.remove(latest_path)
        os.symlink(checkpoint_path, latest_path)


# training
@dataclass
class TrainingConfig:
    image_dm_config: ImageDataModuleConfig
    unet_config: UNetConfig
    consistency_training: ImprovedConsistencyTraining
    consistency_sampling: ConsistencySamplingAndEditing
    lit_icm_config: LitImprovedConsistencyModelConfig
    trainer: Trainer
    seed: int = 42
    model_ckpt_path: str = "checkpoints/"
    resume_ckpt_path: Optional[str] = None

    def __post_init__(self):
        if args.train_continue:
            self.resume_ckpt_path = self._find_last_checkpoint()

    def _find_last_checkpoint(self) -> Optional[str]:
        dataset_name = os.path.basename(self.image_dm_config.data_dir.strip("/"))
        checkpoint_folder = os.path.join(self.model_ckpt_path, dataset_name)
        latest_path = os.path.join(checkpoint_folder, "latest.ckpt")
        if os.path.islink(latest_path):
            return os.readlink(latest_path)
        logging.info(
            f"No checkpoints found for dataset '{dataset_name}', starting training from scratch."
        )
        return None


def run_training(config: TrainingConfig) -> None:
    # Set seed
    seed_everything(config.seed)

    # Create data module
    dm = ImageDataModule(config.image_dm_config)

    dataset_name = os.path.basename(config.image_dm_config.data_dir.strip("/"))

    checkpoint_dir = config.model_ckpt_path
    unet_config = config.unet_config
    model = UNet(config.unet_config, os.path.join(checkpoint_dir, dataset_name))
    ema_model = UNet(config.unet_config, os.path.join(checkpoint_dir, dataset_name))
    ema_model.load_state_dict(model.state_dict())

    if config.resume_ckpt_path:
        checkpoint = torch.load(
            config.resume_ckpt_path, map_location=torch.device("cpu")
        )
        state_dict = checkpoint[
            "state_dict"
        ]  # Access the state_dict key within the checkpoint
        # Remove 'model.' prefix
        adapted_state_dict = {
            k[len("model.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        # Now try loading the adapted state dict
        model.load_state_dict(adapted_state_dict)
        # model.load_state_dict(state_dict)
        ema_model.load_state_dict(adapted_state_dict)
    else:
        logging.info("No checkpoint specified, starting training from scratch.")

    # Create lightning module
    lit_icm = LitImprovedConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        unet_config,
        config.lit_icm_config,
        dataset_name,  # Pass dataset name here
        checkpoint_dir,  # And checkpoint directory her
    )
    # Run training
    config.trainer.fit(lit_icm, dm, ckpt_path=config.resume_ckpt_path)

    # Save model
    final_checkpoint_folder = os.path.join(config.model_ckpt_path, dataset_name)

    # Debugging: Print the final checkpoint folder path
    print("Final checkpoint folder path:", final_checkpoint_folder)
    final_model_path = os.path.join(final_checkpoint_folder, "final_model.ckpt")

    # Save the final model
    lit_icm.model.save_pretrained(final_model_path)


training_config = TrainingConfig(
    image_dm_config=ImageDataModuleConfig(
        data_dir=args.data_dir,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    ),
    model_ckpt_path="checkpoints",
    unet_config=UNetConfig(),
    consistency_training=ImprovedConsistencyTraining(final_timesteps=1280),
    consistency_sampling=ConsistencySamplingAndEditing(),
    lit_icm_config=LitImprovedConsistencyModelConfig(
        sample_every_n_steps=args.sample_every_n_steps, lr_scheduler_iters=args.lr_scheduler_iters
    ),
    trainer=Trainer(
        max_steps=args.max_steps,
        precision="16-mixed",
        log_every_n_steps=100,
        logger=TensorBoardLogger(".", name="logs"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
        accelerator="gpu",
        devices=args.devices,
    ),
)

run_training(training_config)
