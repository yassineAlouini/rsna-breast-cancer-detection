import modal






import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader
import torch.optim as optim
from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from pathlib import Path
from torch.nn import Flatten
import modal
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from box import Box
from rsna.metric import pfbeta_torch_thresh, pfbeta_torch
import wandb
from fastai.data.all import *



stub = modal.Stub(name="train-rsna")
image = modal.Image.debian_slim().run_commands(
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "apt-get update",
    "apt install unzip",
    "pip install kaggle",
    "mkdir -p ~/.kaggle/",
    'echo \'{"username":"yassinealouini", "key":"63ead6c78a5527a8d12f6a3ca41e3c48"}\' >> test.json',
    "cp test.json  ~/.kaggle/kaggle.json",
    "chmod 600 ~/.kaggle/kaggle.json",
    # Download data
    "kaggle datasets download -d awsaf49/rsna-bcd-roi-1024x-png-dataset",
    # Unzip
    "unzip rsna-bcd-roi-1024x-png-dataset.zip -d 1024_data"
).pip_install(
    "torch",
    "torchvision",
    "pytorch-lightning",
    "wandb~=0.13.4",
    "albumentations",
    "pandas",
    "timm"
).run_commands("export GIT_PYTHON_REFRESH=quiet", 
               "apt-get install git -y",
               "mv 1024_data ~/1024_data")
volume = modal.SharedVolume().persist("train-rsna")

USE_GPU = "any"

# TODO: Why is this the optimal threshold? Most likely not...
THRESHOLD = 0.402
DEBUG = False
MAX_EPOCHS=2

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
    # TODO: How big is the s model?
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

        return image, y




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
            albu.Resize(1024, 1024),
            albu.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ],
        p=1,
    )

transforms = get_transfos()


class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.softmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        

# TODO: How to fix the F1 wrong metric computation? Always 0....
class BreastCancerModel(pl.LightningModule):

    def __init__(self, model_name, num_classes, batch_size, learning_rate, len_train):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.len_train = len_train
        # TODO: How was this weight computed? axis=-1 to flatten?
        # self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]).float(), reduction="mean")
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([1.0]))
        self.backbone = timm.create_model(model_name, pretrained=True, drop_rate = 0.2, drop_path_rate = 0.0)
        final_in_features = self.backbone.classifier.in_features
        self.head = nn.Sequential(SelectAdaptivePool2d(pool_type='avg', flatten=Flatten()),
                                  nn.Linear(final_in_features, self.num_classes))

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.head(x)
        return x

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

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # forward pass
        y_hat = self(x)
        # calculate loss
        loss = self.loss_fn(y_hat, y)
        # log loss to tensorboard
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        # calculate F1 but probabilistic
        y_hat = torch.sigmoid(y_hat)
        score = pfbeta_torch(y_hat, y)
        thresh_score = pfbeta_torch_thresh(y_hat, y)
        self.log('val_pfbeta', score, on_epoch=True, prog_bar=True)
        self.log('val_thresh_pfbeta', thresh_score, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_pfbeta': score, 'val_thresh_pfbeta': thresh_score}

    def validation_epoch_end(self, outputs):
        # calculate mean loss and accuracy across all validation batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_pfbeta'] for x in outputs]).mean()
        avg_thresh_f1 = torch.stack([x['val_thresh_pfbeta'] for x in outputs]).mean()
        # log mean loss and accuracy to tensorboard
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('val_pfbeta', avg_f1, on_epoch=True, prog_bar=True)
        self.log('val_thresh_pfbeta', avg_thresh_f1, on_epoch=True, prog_bar=True)
        return {'val_loss': avg_loss, 'val_pfbeta': avg_f1, 'val_thresh_pfbeta': avg_thresh_f1}


    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=3e-4, betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3e-4,
            epochs=10,
            steps_per_epoch=int(self.len_train / self.batch_size),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=1.0,
            final_div_factor=10_000.0,
        )
        return [optimizer], [scheduler]


class BreastCancerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 16, fold=0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        df = pd.read_csv(data_dir + "sgkf_train.csv")

        df['path'] = data_dir + "train_images/" + df["patient_id"].astype(str) + "/" + df["image_id"].astype(str) + ".png"
        # Filter using the fold.
        train_df = df[df["fold"] != fold]
        val_df = df[df["fold"] == fold]
        self.train_df = train_df
        self.val_df = val_df

    def setup(self):
        self.breast_train = BreastDataset(self.train_df, transforms)
        self.breast_val = BreastDataset(self.val_df, transforms)

    def train_dataloader(self):
        return DataLoader(self.breast_train, batch_size=self.batch_size, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.breast_val, batch_size=self.batch_size, num_workers=32, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.breast_val, batch_size=self.batch_size)


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

# https://www.youtube.com/watch?v=hOkQSR33yLY&list=PLvc14zohWxi3gTNT_deSJY2LOksaY_fn_&index=2
def train():
    seed_everything(42, workers=True)
    # sets seeds for numpy, torch and python.random.
    for fold in range(4):
        print(f"====== Fold: {fold} ======")
        batch_size = 12
        num_classes = 1
        wandb_logger = WandbLogger(
            project='rsna-breast-cancer', 
            job_type='train', 
            config=class2dict(Config)
        )
        model_name = Config.name
        logger = TensorBoardLogger("/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/lightning_logs/", 
                                   name="rsna-breast-cancer")

        checkpoint_callback = ModelCheckpoint(
            dirpath="/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/checkpoints/",
            filename= f"fold_{fold}_{model_name}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        data_dir = "/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/"
        dm = BreastCancerDataModule(data_dir=data_dir, batch_size=batch_size)
        dm.setup()
        model = BreastCancerModel(model_name=model_name, learning_rate=3e-4, num_classes=num_classes, 
                                  batch_size=batch_size, len_train=len(dm.breast_train))
        val_samples = next(iter(dm.val_dataloader()))


        trainer = Trainer(accelerator="gpu", devices=1,
                  logger=wandb_logger,
                  callbacks=[checkpoint_callback, early_stopping_callback, ImagePredictionLogger(val_samples, num_samples=batch_size)],
                  max_epochs=MAX_EPOCHS,
                  progress_bar_refresh_rate=5,
                  precision=16,
                  fast_dev_run=DEBUG)

        trainer.fit(model, dm)



@stub.function(
    image=image,
    gpu=USE_GPU,
    # secret=modal.Secret.from_name("wandb"),
    # TODO: Probably more...
    timeout=2 * 3600,  # 2 hours maybe more...
    mounts=[]
)
def modal_train():
    seed_everything(42, workers=True)
    # sets seeds for numpy, torch and python.random.
    model = BreastCancerModel(learning_rate=3e-4, num_classes=1, batch_size=16)
    trainer = Trainer(deterministic=True, accelerator="gpu")
    dm = BreastCancerDataModule(data_dir="/root/1024_data/")
    trainer.fit(model, dm)


if __name__ == "__main__":
    # with stub.run():
    #     modal_train()
    train()