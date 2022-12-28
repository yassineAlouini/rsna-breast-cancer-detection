# import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningModule
import torch
import pandas as pd
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import pytorch_lightning as pl
import timm


DATA_FOLDER = "/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/"

# From https://www.kaggle.com/code/theoviel/rsna-breast-baseline-inference
class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    device = "cuda"
    save_weights = True

    # Images
    size = 1024

    # k-fold
    k = 4  # Stratified GKF

    # Model
    name = "tf_efficientnetv2_s"
    pretrained_weights = None
    # Cancer or not cancer (so binary predictions)
    num_classes = 1
    n_channels = 3

    # Training    
    loss_config = {
        "name": "bce",
        "smoothing": 0.,
        "activation": "sigmoid",
    }

    data_config = {
        "batch_size": 8,
        "val_bs": 8,
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 3e-4,
        "warmup_prop": 0.1,
        "betas": (0.9, 0.999),
        "max_grad_norm": 10.,
    }

    epochs = 4
    use_fp16 = True
    
    ## Other stuff
    # Augmentations : Only HorizontalFlip

"""
-> 41092 training images
-> 13614 validation images
-> 21459769 trainable parameters
"""


# Objective: make a first model using efficient net and 512 size images.
# Add wandb logger
# Make an inference


import cv2
import torch
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    """
    Image torch Dataset.
    """
    def __init__(
        self,
        df,
        transforms=None,
    ):
        """
        Constructor

        Args:
            paths (list): Path to images.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.paths = df['path'].values
        self.transforms = transforms
        self.targets = df['cancer'].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """
        image = cv2.imread(self.paths[idx])

        if self.transforms:
            image = self.transforms(image=image)["image"]

        y = torch.tensor([self.targets[idx]], dtype=torch.float)
        # TODO: Probably not needed...
        w = torch.tensor([1])

        return image, y, w




def get_transfos(augment=True, visualize=False):
    """
    Returns transformations.

    Args:
        augment (bool, optional): Whether to apply augmentations. Defaults to True.
        visualize (bool, optional): Whether to use transforms for visualization. Defaults to False.

    Returns:
        albumentation transforms: transforms.
    """
    return albu.Compose(
        [
            albu.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        p=1,
    )

df = pd.read_csv("train.csv")
df['path'] = DATA_FOLDER + df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png"


dataset = BreastDataset(df)
img, target, weight = dataset[0]

print(img.mean(), img.max(), img.min())
# To be continued...



# TODO: Finish...
class BreastCancerModel(pl.LightningModule):

    def __init__(self, num_classes, data_dir, batch_size, learning_rate, 
                 num_groups, group_size):
        super().__init__()
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_groups = num_groups
        self.group_size = group_size

        # define model, loss function, and optimizer
        # TODO: Replace with timm import...
        self.model = enet.EfficientNet.from_name('efficientnet-b0')
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # forward pass
        y_hat = self(x)
        # calculate loss
        loss = self.loss_fn(y_hat, y)
        # log loss to tensorboard
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, group):
        # get data and labels from batch
        x, y = batch
        # forward pass
        y_hat = self(x)
        # calculate loss
        loss = self.loss_fn(y_hat, y)
        # log loss to tensorboard
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, group=group)
        # calculate accuracy
        acc = self.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, group=group)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs, group):
        # calculate mean loss and accuracy across all validation batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        # log mean loss and accuracy to tensorboard
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True, group=group)
        self.log('val_acc', avg_acc, on_epoch=True, prog_bar=True, group=group)
        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def test_step(self, batch, batch_id):
        pass
