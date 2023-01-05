import modal
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
            albu.Resize(1024, 1024),
            ToTensorV2(),
        ],
        p=1,
    )

transforms = get_transfos()



# To be continued...
def pfbeta_torch(labels, predictions, beta=1.0):
    y_true_count = torch.sum(labels)
    ctp = 0
    cfp = 0

    predictions = torch.clamp(predictions, min=0, max=1)
    ctp = torch.sum(predictions * labels)
    cfp = torch.sum(predictions * (1.0-labels))

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


class BreastCancerModel(pl.LightningModule):

    def __init__(self, num_classes, batch_size, learning_rate):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # define model, loss function, and optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()


        # A bit inspired from here: https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images
        # Trying 'efficientnet_b3' for now.
        self.backbone = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)
        final_in_features = self.backbone.classifier.in_features
        self.pooling =  nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(self.num_classes)
        self.head = nn.Sequential(SelectAdaptivePool2d(pool_type='avg', flatten=Flatten()), 
                                  nn.Linear(1280, 1))
        self.fc = nn.Linear(1000, 1)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        return self.head(x)

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
        # calculate F1 but probabilistic
        score = pfbeta_torch(y_hat, y)
        self.log('val_pfbeta', score, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_pfbeta': score}

    # def validation_epoch_end(self, outputs):
    #     # calculate mean loss and accuracy across all validation batches
    #     # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     # avg_acc = torch.stack([x['val_pfbeta'] for x in outputs]).mean()
    #     # log mean loss and accuracy to tensorboard
    #     self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
    #     self.log('val_pfbeta', avg_acc, on_epoch=True, prog_bar=True)
    #     return {'val_loss': avg_loss, 'val_pfbeta': avg_acc}

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
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        df = pd.read_csv(data_dir + "train.csv")
        df['path'] = data_dir + "train_images/" + df["patient_id"].astype(str) + "/" + df["image_id"].astype(str) + ".png"
        self.df = df

    def setup(self, stage: str):
        self.breast_test = BreastDataset(self.df, transforms)
        self.breast_train = BreastDataset(self.df, transforms)

    def train_dataloader(self):
        return DataLoader(self.breast_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.breast_test, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.breast_test, batch_size=self.batch_size)



def train():
    seed_everything(42, workers=True)
    # sets seeds for numpy, torch and python.random.
    for fold in range(4):
        print(f"====== Fold: {fold} ======")
        model = BreastCancerModel(learning_rate=3e-4, num_classes=1, batch_size=8)
        wandb_logger = WandbLogger(
            project='rsna-breast-cancer', 
            job_type='train', 
            config=Config
        )

        logger = TensorBoardLogger("lightning_logs", name="rsna-breast-cancer")

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename= f"fold_{fold}_efficient_net",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min"
        )

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
        trainer = Trainer(accelerator="gpu", devices=1,
                  logger=wandb_logger,
                  callbacks=[checkpoint_callback, early_stopping_callback],
                  max_epochs=10,
                  progress_bar_refresh_rate=30,
                  precision=16)
        data_dir = "/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/"
        dm = BreastCancerDataModule(data_dir=data_dir, batch_size=8)
        trainer.fit(model, dm)



# TODO: Make it a modal thing...
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
    # TODO: Add wandb monitoring, fix loss...