from fastai.vision.learner import *
from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import ActivationType
from fastai.callback.wandb import *

from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import pandas as pd
import numpy as np
from pdb import set_trace
import wandb

NUM_EPOCHS = 4
NUM_SPLITS = 4
BATCH_SIZE = 16
RESIZE_TO = (1024, 1024)
SEED = 42

label_smoothing_weights = torch.tensor([1,10]).float()
if torch.cuda.is_available():
    label_smoothing_weights = label_smoothing_weights.cuda()

## Creating stratified splits for training

DATA_PATH = "/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/"
TRAIN_IMAGE_DIR = "/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/"
MODEL_PATH = '/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/models/'
train_csv = pd.read_csv(f'{DATA_PATH}/train.csv')
patient_id_any_cancer = train_csv.groupby('patient_id').cancer.max().reset_index()
skf = StratifiedKFold(NUM_SPLITS, shuffle=True, random_state=SEED)
splits = list(skf.split(patient_id_any_cancer.patient_id, patient_id_any_cancer.cancer))

# TODO: Use the knowledge from here to fix the PyTorch-Lightning pipeline...

wandb.init(project='rsna-breast-cancer')
MODEL_NAME = 'tf_efficientnetv2_b2'

#https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369267  
def pfbeta_torch(preds, labels, beta=1):
    if preds.dim() != 2 or (preds.dim() == 2 and preds.shape[1] !=2): raise ValueError('Houston, we got a problem')
    preds = preds[:, 1]
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0.0

# https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/369886    
def pfbeta_torch_thresh(preds, labels):
    optimized_preds = optimize_preds(preds, labels)
    return pfbeta_torch(optimized_preds, labels)

def optimize_preds(preds, labels=None, thresh=None, return_thresh=False, print_results=False):
    preds = preds.clone()
    if labels is not None: without_thresh = pfbeta_torch(preds, labels)
    
    if not thresh and labels is not None:
        threshs = np.linspace(0, 1, 101)
        f1s = [pfbeta_torch((preds > thr).float(), labels) for thr in threshs]
        idx = np.argmax(f1s)
        thresh, best_pfbeta = threshs[idx], f1s[idx]

    preds = (preds > thresh).float()

    if print_results:
        print(f'without optimization: {without_thresh}')
        pfbeta = pfbeta_torch(preds, labels)
        print(f'with optimization: {pfbeta}')
        print(f'best_thresh = {thresh}')
    if return_thresh:
        return thresh
    return preds

fn2label = {fn: cancer_or_not for fn, cancer_or_not in zip(train_csv['image_id'].astype('str'), train_csv['cancer'])}

def splitting_func(paths):
    train = []
    valid = []
    for idx, path in enumerate(paths):
        if int(path.parent.name) in patient_id_any_cancer.iloc[splits[SPLIT][0]].patient_id.values:
            train.append(idx)
        else:
            valid.append(idx)
    return train, valid

def label_func(path):
    return fn2label.get(path.stem, 0)

def get_items(image_dir_path):
    items = []
    for p in get_image_files(image_dir_path):
        items.append(p)
        if p.stem in fn2label and int(p.parent.name) in patient_id_any_cancer.iloc[splits[SPLIT][0]].patient_id.values:
            if label_func(p) == 1:
                for _ in range(5):
                    items.append(p)
    return items

from timm.models.layers.adaptive_avgmax_pool import SelectAdaptivePool2d
from torch.nn import Flatten


def get_dataloaders():
    train_image_path = TRAIN_IMAGE_DIR

    dblock = DataBlock(
        blocks    = (ImageBlock, CategoryBlock),
        get_items = get_items,
        get_y = label_func,
        splitter  = splitting_func,
        batch_tfms=[Flip()],
        item_tfms=Resize((1024, 1024))
    )
    dsets = dblock.datasets(train_image_path)
    return dblock.dataloaders(train_image_path, batch_size=BATCH_SIZE)

def get_learner(arch=resnet18):
    learner = vision_learner(
        get_dataloaders(),
        arch,
        # 512 is for ResNet18
        # 1408 for b2
        custom_head=nn.Sequential(SelectAdaptivePool2d(pool_type='avg', flatten=Flatten()), 
                                  nn.Linear(1408, 2)),
        metrics=[
            error_rate,
            AccumMetric(pfbeta_torch, activation=ActivationType.Softmax, flatten=False),
            AccumMetric(pfbeta_torch_thresh, activation=ActivationType.Softmax, flatten=False)
        ],
        loss_func=CrossEntropyLossFlat(weight=torch.tensor([1,50]).float()),
        pretrained=True,
        cbs=WandbCallback(),
        normalize=False
    ).to_fp16()
    return learner

preds, labels = [], []

SPLIT = 0 # our learner needs this to construct its dataloaders...
learn = get_learner(MODEL_NAME)

for SPLIT in range(NUM_SPLITS):
    learn = get_learner(MODEL_NAME)
    learn.unfreeze()
    learn.fit_one_cycle(NUM_EPOCHS, 1e-4, pct_start=0.1)
    learn.save(f'{MODEL_PATH}/{SPLIT}_{MODEL_NAME}')
        
    output = learn.get_preds()
    preds.append(output[0])
    labels.append(output[1])

threshold = optimize_preds(torch.cat(preds), torch.cat(labels), return_thresh=True, print_results=True)
print(threshold)

"""
without optimization: 0.06475809961557388                                                                                                   
with optimization: 0.1671641767024994
best_thresh = 0.96
0.96
"""



