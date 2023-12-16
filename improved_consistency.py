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

viz = Visdom(env="consistency_model_anime_face_lightning")
viz.line([0.0], [0.0], win="consis_loss", opts=dict(title="loss over time"))

from options import parse_opts

args = parse_opts()
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


# Modules
def GroupNorm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, channels // 4), num_channels=channels)


class SelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.dropout = dropout

        self.qkv_projection = nn.Sequential(
            GroupNorm(in_channels),
            nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, bias=False),
            Rearrange("b (i h d) x y -> i b h (x y) d", i=3, h=n_heads),
        )
        self.output_projection = nn.Sequential(
            Rearrange("b h l d -> b l (h d)"),
            nn.Linear(in_channels, out_channels, bias=False),
            Rearrange("b l d -> b d l"),
            GroupNorm(out_channels),
            nn.Dropout1d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = self.qkv_projection(x).unbind(dim=0)

        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        output = self.output_projection(output)
        output = rearrange(output, "b c (x y) -> b c x y", x=x.shape[-2], y=x.shape[-1])

        return output + self.residual_projection(x)


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.noise_level_projection = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(noise_level_channels, out_channels, kernel_size=1),
        )
        self.output_projection = nn.Sequential(
            GroupNorm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.Dropout2d(dropout),
        )
        self.residual_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        h = self.input_projection(x)
        h = h + self.noise_level_projection(noise_level)

        return self.output_projection(h) + self.residual_projection(x)


class UNetBlockWithSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        n_heads: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.unet_block = UNetBlock(
            in_channels, out_channels, noise_level_channels, dropout
        )
        self.self_attention = SelfAttention(
            out_channels, out_channels, n_heads, dropout
        )

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        return self.self_attention(self.unet_block(x, noise_level))


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            Rearrange("b c (h ph) (w pw) -> b (c ph pw) h w", ph=2, pw=2),
            nn.Conv2d(4 * channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding="same"),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class NoiseLevelEmbedding(nn.Module):
    def __init__(self, channels: int, scale: float = 0.02) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(channels // 2) * scale, requires_grad=False)

        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
            Rearrange("b c -> b c () ()"),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = x[:, None] * self.W[None, :] * 2 * torch.pi
        h = torch.cat([torch.sin(h), torch.cos(h)], dim=-1)

        return self.projection(h)


# Unet
@dataclass
class UNetConfig:
    channels: int = 3
    noise_level_channels: int = 256
    noise_level_scale: float = 0.02
    n_heads: int = 8
    top_blocks_channels: Tuple[int, ...] = (128, 128)
    top_blocks_n_blocks_per_resolution: Tuple[int, ...] = (2, 2)
    top_blocks_has_resampling: Tuple[bool, ...] = (True, True)
    top_blocks_dropout: Tuple[float, ...] = (0.0, 0.0)
    mid_blocks_channels: Tuple[int, ...] = (256, 512)
    mid_blocks_n_blocks_per_resolution: Tuple[int, ...] = (4, 4)
    mid_blocks_has_resampling: Tuple[bool, ...] = (True, False)
    mid_blocks_dropout: Tuple[float, ...] = (0.0, 0.3)


class UNet(nn.Module):
    def __init__(self, config: UNetConfig, checkpoint_dir: str = "") -> None:
        super().__init__()

        self.config = config
        self.checkpoint_dir = checkpoint_dir

        self.input_projection = nn.Conv2d(
            config.channels,
            config.top_blocks_channels[0],
            kernel_size=3,
            padding="same",
        )
        self.noise_level_embedding = NoiseLevelEmbedding(
            config.noise_level_channels, config.noise_level_scale
        )
        self.top_encoder_blocks = self._make_encoder_blocks(
            self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
            self.config.top_blocks_n_blocks_per_resolution,
            self.config.top_blocks_has_resampling,
            self.config.top_blocks_dropout,
            self._make_top_block,
        )
        self.mid_encoder_blocks = self._make_encoder_blocks(
            self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
            self.config.mid_blocks_n_blocks_per_resolution,
            self.config.mid_blocks_has_resampling,
            self.config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.mid_decoder_blocks = self._make_decoder_blocks(
            self.config.mid_blocks_channels + self.config.mid_blocks_channels[-1:],
            self.config.mid_blocks_n_blocks_per_resolution,
            self.config.mid_blocks_has_resampling,
            self.config.mid_blocks_dropout,
            self._make_mid_block,
        )
        self.top_decoder_blocks = self._make_decoder_blocks(
            self.config.top_blocks_channels + self.config.mid_blocks_channels[:1],
            self.config.top_blocks_n_blocks_per_resolution,
            self.config.top_blocks_has_resampling,
            self.config.top_blocks_dropout,
            self._make_top_block,
        )
        self.output_projection = nn.Conv2d(
            config.top_blocks_channels[0],
            config.channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        h = self.input_projection(x)
        noise_level = self.noise_level_embedding(noise_level)

        top_encoder_embeddings = []
        for block in self.top_encoder_blocks:
            if isinstance(block, UNetBlock):
                h = block(h, noise_level)
                top_encoder_embeddings.append(h)
            else:
                h = block(h)

        mid_encoder_embeddings = []
        for block in self.mid_encoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = block(h, noise_level)
                mid_encoder_embeddings.append(h)
            else:
                h = block(h)

        for block in self.mid_decoder_blocks:
            if isinstance(block, UNetBlockWithSelfAttention):
                h = torch.cat((h, mid_encoder_embeddings.pop()), dim=1)
                h = block(h, noise_level)
            else:
                h = block(h)

        for block in self.top_decoder_blocks:
            if isinstance(block, UNetBlock):
                h = torch.cat((h, top_encoder_embeddings.pop()), dim=1)
                h = block(h, noise_level)
            else:
                h = block(h)

        return self.output_projection(h)

    def _make_encoder_blocks(
        self,
        channels: Tuple[int, ...],
        n_blocks_per_resolution: Tuple[int, ...],
        has_resampling: Tuple[bool, ...],
        dropout: Tuple[float, ...],
        block_fn: Callable[[], nn.Module],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()

        channel_pairs = list(zip(channels[:-1], channels[1:]))
        for idx, (in_channels, out_channels) in enumerate(channel_pairs):
            for _ in range(n_blocks_per_resolution[idx]):
                blocks.append(block_fn(in_channels, out_channels, dropout[idx]))
                in_channels = out_channels

            if has_resampling[idx]:
                blocks.append(Downsample(out_channels))

        return blocks

    def _make_decoder_blocks(
        self,
        channels: Tuple[int, ...],
        n_blocks_per_resolution: Tuple[int, ...],
        has_resampling: Tuple[bool, ...],
        dropout: Tuple[float, ...],
        block_fn: Callable[[], nn.Module],
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()

        channel_pairs = list(zip(channels[:-1], channels[1:]))[::-1]
        for idx, (out_channels, in_channels) in enumerate(channel_pairs):
            if has_resampling[::-1][idx]:
                blocks.append(Upsample(in_channels))

            inner_blocks = []
            for _ in range(n_blocks_per_resolution[::-1][idx]):
                inner_blocks.append(
                    block_fn(in_channels * 2, out_channels, dropout[::-1][idx])
                )
                out_channels = in_channels
            blocks.extend(inner_blocks[::-1])

        return blocks

    def _make_top_block(
        self, in_channels: int, out_channels: int, dropout: float
    ) -> UNetBlock:
        return UNetBlock(
            in_channels,
            out_channels,
            self.config.noise_level_channels,
            dropout,
        )

    def _make_mid_block(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> UNetBlockWithSelfAttention:
        return UNetBlockWithSelfAttention(
            in_channels,
            out_channels,
            self.config.noise_level_channels,
            self.config.n_heads,
            dropout,
        )

    def save_pretrained(self, full_model_path: str):
        # Extract the directory from the full model path
        model_dir = os.path.dirname(full_model_path)

        # Ensure the directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        torch.save(self.state_dict(), full_model_path)

        # Save the configuration alongside the model
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "UNet":
        # Load the configuration file
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
        config = UNetConfig(**config_dict)  # Create the config object

        # Instantiate the model with the loaded configuration
        model = cls(config)

        # Load the model's state dictionary
        model_path = os.path.join(model_dir, "final_model.ckpt")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        return model


# summary(UNet(UNetConfig()), input_size=((1, 3, 32, 32), (1,)))

# LitUNet
@dataclass
class LitImprovedConsistencyModelConfig:
    ema_decay_rate: float = 0.99993
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10_000
    sample_every_n_steps: int = 1500
    num_samples: int = 32
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
        sample_every_n_steps=args.sample_every_n_steps, lr_scheduler_iters=1000
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
