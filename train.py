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
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d


# Get this from Kaggle infra
if LOCAL:
    DATA_FOLDER = "/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/"

if CLOUD:
    # Some data to try the pipeline
    DATA_FOLDER = "1024_data/"



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
    name = "efficientnet_b2"
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

transforms = get_transfos()

df = pd.read_csv("train.csv")
df['path'] = DATA_FOLDER + df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png"

print(df.head())

# To be continued...



# TODO: Finish...
class BreastCancerModel(pl.LightningModule):

    def __init__(self, num_classes, batch_size, learning_rate):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # define model, loss function, and optimizer
        self.loss_fn = nn.CrossEntropyLoss()


        # A bit inspired from here: https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images
        # Trying 'efficientnet_b3' for now.
        self.backbone = timm.create_model("efficientnet_b2", pretrained=True)
        final_in_features = self.backbone.classifier.in_features
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(final_in_features, self.num_classes)
        self.bn = nn.BatchNorm1d(self.num_classes)


        self.optimizer = optim.Adam(self.backbone.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        print(x.shape)
        x = self.pooling(x).view(batch_size, -1)
        x = self.dropout(x)
        print(x.shape)
        x = self.fc(x)
        x = self.bn(x)

    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y, _ = batch
        # forward pass
        y_hat = self(x)
        # calculate loss
        loss = self.loss_fn(y_hat, y)
        # log loss to tensorboard
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y, _ = batch
        # forward pass
        y_hat = self(x)
        # calculate loss
        loss = self.loss_fn(y_hat, y)
        # log loss to tensorboard
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        # calculate accuracy
        acc = self.accuracy(y_hat, y)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        # calculate mean loss and accuracy across all validation batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        # log mean loss and accuracy to tensorboard
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', avg_acc, on_epoch=True, prog_bar=True)
        return {'val_loss': avg_loss, 'val_acc': avg_acc}

    def test_step(self, batch, batch_id):
        pass


    def configure_optimizers(self):
        optimizer = eval("optim.AdamW")(
            self.parameters(), lr=3e-4, betas=(0.9, 0.999),
        )
        # scheduler = eval(self.cfg.scheduler.name)(
        #     optimizer,
        #     **self.cfg.scheduler.params
        # )
        return [optimizer]


class BreastCancerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.breast_test = BreastDataset(df, transforms)
        self.breast_train = BreastDataset(df, transforms)

    def train_dataloader(self):
        return DataLoader(self.breast_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.breast_test, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.breast_test, batch_size=self.batch_size)



seed_everything(42, workers=True)
# sets seeds for numpy, torch and python.random.
model = BreastCancerModel(learning_rate=3e-4, num_classes=1, batch_size=32)
# trainer = Trainer(deterministic=True, accelerator="gpu", devices=1)
trainer = Trainer(deterministic=True)
dm = BreastCancerDataModule()
trainer.fit(model, dm)