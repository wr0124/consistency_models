from typing import Tuple, List, Union
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataclasses import dataclass
from torch import Tensor
from options import parse_opts
from UNet import UNet, UNetConfig
import torch.optim as optim
from consistency_models.consistency_models import ImprovedConsistencyTraining, pseudo_huber_loss, ConsistencySamplingAndEditing
from torchvision.utils import make_grid
from visdom import Visdom
import torchvision.utils as vutils
from torchinfo import summary
import numpy as np
import random
import torch
import os
from torchvision.datasets.folder import default_loader
import glob
# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
args = parse_opts()
viz = Visdom(env=args.env)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_value = 42  
set_seed(seed_value)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None,
                 image_folder="img", mask_folder="bbox"):
        super(CustomImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform,
            loader=loader, is_valid_file=is_valid_file
        )
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def __getitem__(self, index):
        file_path, _ = self.samples[index]
       # print(f"filename is {file_path}")
        filename = os.path.basename(file_path)
        image_path = os.path.join(self.root, self.image_folder, filename)
        mask_path = os.path.join(self.root, self.mask_folder, filename)

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or mask not found for file: {filename}")

        image = self.loader(image_path)
        mask = self.loader(mask_path)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Data module configuration
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True

# Data module class
class ImageDataModule:
    def __init__(self, config: ImageDataModuleConfig) -> None:
        self.config = config
        self.dataset = None

    def setup(self) -> None:
        transform = T.Compose([
            T.Resize(self.config.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
        ])
        self.dataset = CustomImageFolder(
            self.config.data_dir,
            transform=transform,
            image_folder=args.data_dir+"/img/",  # Specify the folder for images
            mask_folder=args.data_dir+"/bbox/" ,    # Specify the folder for masks
        )


    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )

# Extract the dataset name from the data_dir
dataset_name = os.path.basename(args.data_dir)
checkpoint_dir = os.path.join("checkpoints", dataset_name)
print(f"checkpoint_dir is {checkpoint_dir}")


model = UNet(UNetConfig())
# Load the model checkpoint
checkpoint_path = os.path.join(checkpoint_dir, "final_model.ckpt")
checkpoint = torch.load(checkpoint_path, map_location=device)
if 'state_dict' in checkpoint:
    # If 'state_dict' is a key in the checkpoint, use it directly
    model.load_state_dict(checkpoint['state_dict'])
else:
    # If 'state_dict' is not a key, assume the checkpoint is the state_dict
    model.load_state_dict(checkpoint)

model.eval()

test_data_config = ImageDataModuleConfig(
    data_dir=args.data_dir,
    image_size=args.image_size,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
test_data_module = ImageDataModule(test_data_config)
test_data_module.setup()
test_dataloader = test_data_module.get_dataloader(shuffle=False)

consistency_sampling=ConsistencySamplingAndEditing()    
sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )

def __log_images(
        images: Tensor,
        title: str,
        window_name: str,
        window_size: Tuple[int, int],
    ) -> None:
        images = images.detach().float()

        grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True, nrow=8
        )

        grid = grid.cpu().numpy()
        if grid.min() < 0:
            grid = (grid + 1) / 4
        width, height = window_size
        viz.images(
            grid,
            nrow=4,
            opts=dict(
                title=f"{title}",
                caption=f"{title}",
                width=width,
                height=height,
            ),
            win=window_name,
        )
  
for batch_idx, (images, masks) in enumerate(test_dataloader):
    # Transfer data to the device
    images, masks = images.to(device), masks.to(device)
    binary_masks = (masks != -1).float() 
    
    viz.images(
            vutils.make_grid(
                images.to(dtype=torch.float32), normalize=True, value_range=(-1,1), nrow=int(images.shape[0] / 4)
            ),
            win="consistency_test3",
            opts=dict(
                title="input image",
                caption="input image",
                width=500,
                height=500,
            ),
        )
    viz.images(
            vutils.make_grid(
                binary_masks.to(dtype=torch.float32), normalize=True, value_range=(-1,1), nrow=int(images.shape[0] / 4)
            ),
            win="consistency_test4",
            opts=dict(
                title="mask image",
                caption="mask image",
                width=500,
                height=500,
            ),
        )


    # Apply consistency sampling and editing
    for sigmas in sampling_sigmas:
        print( f"type images {type(images)} and shape {images.shape}")
        print( f"binary_masks images {type(binary_masks)} and shape {binary_masks.shape}")
        print( f"type sigmas {type(sigmas)} and shape {sigmas}")
        edited_samples = consistency_sampling(
                model.to(device), images.to(device), sigmas, binary_masks.to(device),  clip_denoised=True,verbose=True )
        __log_images(
                edited_samples,
                f"generated_samples_window_{sigmas}",
                window_name=f"generated_samples_window_{sigmas}",
                window_size=(400, 400),
            )
