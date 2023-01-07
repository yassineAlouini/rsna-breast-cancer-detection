import os
import torch
import random
import numpy as np


def load_model_weights(model, filename, verbose=1, cp_folder="", strict=True):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities.

    Args:
        model (torch model): Model to load the weights to.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to load from. Defaults to "".

    Returns:
        torch model: Model with loaded weights.
    """
    state_dict = torch.load(os.path.join(cp_folder, filename), map_location="cpu")

    try:
        model.load_state_dict(state_dict, strict=strict)
    except BaseException:
        try:
            del state_dict['logits.weight'], state_dict['logits.bias']
            model.load_state_dict(state_dict, strict=strict)
        except BaseException:
            del state_dict['encoder.conv_stem.weight']
            model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(f"\n -> Loading encoder weights from {os.path.join(cp_folder,filename)}\n")

    return model

import torch
import numpy as np
from torch.utils.data import DataLoader

NUM_WORKERS = 2


def predict(model, dataset, loss_config, batch_size=64, device="cuda"):
    """
    Torch predict function.

    Args:
        model (torch model): Model to predict with.
        dataset (CustomDataset): Dataset to predict on.
        loss_config (dict): Loss config, used for activation functions.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()
    preds = np.empty((0,  model.num_classes))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)

            # Forward
            pred, pred_aux = model(x)

            # Get probabilities
            if loss_config['activation'] == "sigmoid":
                pred = pred.sigmoid()
            elif loss_config['activation'] == "softmax":
                pred = pred.softmax(-1)

            preds = np.concatenate([preds, pred.cpu().numpy()])

    return preds



import sys
sys.path.append('/kaggle/input/timm-0-6-9/pytorch-image-models-master')

import timm
import torch
import torch.nn as nn


def define_model(
    name,
    num_classes=1,
    num_classes_aux=0,
    n_channels=1,
    pretrained_weights="",
    pretrained=True,
):
    """
    Loads a pretrained model & builds the architecture.
    Supports timm models.

    Args:
        name (str): Model name
        num_classes (int, optional): Number of classes. Defaults to 1.
        num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
        n_channels (int, optional): Number of image channels. Defaults to 3.
        pretrained_weights (str, optional): Path to pretrained encoder weights. Defaults to ''.
        pretrained (bool, optional): Whether to load timm pretrained weights.

    Returns:
        torch model -- Pretrained model.
    """
    # Load pretrained model
    encoder = getattr(timm.models, name)(pretrained=pretrained)
    encoder.name = name

    # Tile Model
    model = ClsModel(
        encoder,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
    )

    if pretrained_weights:
        model = load_model_weights(model, pretrained_weights, verbose=1, strict=False)

    return model


class ClsModel(nn.Module):
    """
    Model with an attention mechanism.
    """
    def __init__(
        self,
        encoder,
        num_classes=1,
        num_classes_aux=0,
        n_channels=3,
    ):
        """
        Constructor.

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()

        self.encoder = encoder
        self.nb_ft = encoder.num_features

        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.n_channels = n_channels

        self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        self._update_num_channels()

    def _update_num_channels(self):
        if self.n_channels != 3:
            for n, m in self.encoder.named_modules():
                if n:
                    # print("Replacing", n)
                    old_conv = getattr(self.encoder, n)
                    new_conv = nn.Conv2d(
                        self.n_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None,
                    )
                    setattr(self.encoder, n, new_conv)
                    break

    def extract_features(self, x):
        """
        Extract features function.

        Args:
            x (torch tensor [batch_size x 3 x w x h]): Input batch.

        Returns:
            torch tensor [batch_size x num_features]: Features.
        """
        fts = self.encoder.forward_features(x)

        while len(fts.size()) > 2:
            fts = fts.mean(-1)

        return fts

    def get_logits(self, fts):
        """
        Computes logits.

        Args:
            fts (torch tensor [batch_size x num_features]): Features.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        logits = self.logits(fts)

        if self.num_classes_aux:
            logits_aux = self.logits_aux(fts)
        else:
            logits_aux = torch.zeros((fts.size(0)))

        return logits, logits_aux

    def forward(self, x, return_fts=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x n_frames x h x w]): Input batch.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        fts = self.extract_features(x)

        logits, logits_aux = self.get_logits(fts)

        return logits, logits_aux

    



df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")
df['cancer'] = 0

if DEBUG:
    df = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
    df['path'] = SAVE_FOLDER + df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png"
    df = df[df['path'].apply(lambda x: os.path.exists(x))].reset_index(drop=True)

df['path'] = SAVE_FOLDER + df["patient_id"].astype(str) + "_" + df["image_id"].astype(str) + ".png"




USE_TTA = False
predict_fct = predict_tta if USE_TTA else predict

EXP_FOLDERS = [
    "/kaggle/input/rsna-breast-weights-public/tf_efficientnetv2_s_1024/"
]


all_preds = []
for exp_folder in EXP_FOLDERS:
    config = Config

    model = define_model(
        config.name,
        num_classes=config.num_classes,
        num_classes_aux=0,
        n_channels=3,
        pretrained=False
    )
    model = model.cuda().eval()

    dataset = BreastDataset(
        df,
        transforms=get_transfos(augment=False),
    )
    
    weights = sorted(glob.glob(exp_folder + f"*.pt"))
    if not len(weights):
        print('No weights found, add your own dataset !')

    preds = []
    for fold, weight in enumerate(weights):
        model = load_model_weights(model, weight, verbose=1)
        pred = predict(model, dataset, config.loss_config, batch_size=8)
        preds.append(pred)

    preds = np.mean(preds, 0)
    all_preds.append(preds)
    
preds_blend = np.mean(all_preds, 0)


THRESHOLD = 0.2

df["cancer"] = preds_blend
df['prediction_id'] = df['patient_id'].astype(str) + "_" + df['laterality']

sub = df[['prediction_id', 'cancer']].groupby("prediction_id").mean().reset_index()
sub["cancer"] = (sub["cancer"] > THRESHOLD).astype(int)

sub.to_csv('/kaggle/working/submission.csv', index=False)

sub.head()


IMG_PATH = "/kaggle/input/rsna-breast-cancer-detection/test_images/"
test_images = glob.glob(f"{IMG_PATH}*/*.dcm")

if DEBUG:
    IMG_PATH = "/kaggle/input/rsna-breast-cancer-detection/train_images/"
#     test_images = glob.glob(f"{IMG_PATH}*/*.dcm")[:1000]
    test_images = glob.glob(f"{IMG_PATH}10042/*.dcm")
    
print("Number of images :", len(test_images))




SAVE_FOLDER = "/tmp/output/"
SIZE = 1024

os.makedirs(SAVE_FOLDER, exist_ok=True)

if len(test_images) > 100:
    N_CHUNKS = 4
else:
    N_CHUNKS = 1

CHUNKS = [(len(test_images) / N_CHUNKS * k, len(test_images) / N_CHUNKS * (k + 1)) for k in range(N_CHUNKS)]
CHUNKS = np.array(CHUNKS).astype(int)
    
J2K_FOLDER = "/tmp/j2k/"


import torch
import torch.nn.functional as F
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali import pipeline_def
from nvidia.dali.types import DALIDataType
from pydicom.filebase import DicomBytesIO
from nvidia.dali.plugin.pytorch import feed_ndarray, to_torch_type


def convert_dicom_to_j2k(file, save_folder=""):
    patient = file.split('/')[-2]
    image = file.split('/')[-1][:-4]
    dcmfile = pydicom.dcmread(file)

    if dcmfile.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':
        with open(file, 'rb') as fp:
            raw = DicomBytesIO(fp.read())
            ds = pydicom.dcmread(raw)
        offset = ds.PixelData.find(b"\x00\x00\x00\x0C")  #<---- the jpeg2000 header info we're looking for
        hackedbitstream = bytearray()
        hackedbitstream.extend(ds.PixelData[offset:])
        with open(save_folder + f"{patient}_{image}.jp2", "wb") as binary_file:
            binary_file.write(hackedbitstream)

            
@pipeline_def
def j2k_decode_pipeline(j2kfiles):
    jpegs, _ = fn.readers.file(files=j2kfiles)
    images = fn.experimental.decoders.image(jpegs, device='mixed', output_type=types.ANY_DATA, dtype=DALIDataType.UINT16)
    return images

for chunk in tqdm(CHUNKS):
    os.makedirs(J2K_FOLDER, exist_ok=True)

    _ = Parallel(n_jobs=2)(
        delayed(convert_dicom_to_j2k)(img, save_folder=J2K_FOLDER)
        for img in test_images[chunk[0]: chunk[1]]
    )
    
    j2kfiles = glob.glob(J2K_FOLDER + "*.jp2")

    if not len(j2kfiles):
        continue

    pipe = j2k_decode_pipeline(j2kfiles, batch_size=1, num_threads=2, device_id=0, debug=True)
    pipe.build()

    for i, f in enumerate(j2kfiles):
        patient, image = f.split('/')[-1][:-4].split('_')
        dicom = pydicom.dcmread(IMG_PATH + f"{patient}/{image}.dcm")

        out = pipe.run()

        # Dali -> Torch
        img = out[0][0]
        img_torch = torch.empty(img.shape(), dtype=torch.int16, device="cuda")
        feed_ndarray(img, img_torch, cuda_stream=torch.cuda.current_stream(device=0))
        img = img_torch.float()

        # Scale, resize, invert on GPU !
        min_, max_ = img.min(), img.max()
        img = (img - min_) / (max_ - min_)

        if SIZE:
            img = F.interpolate(img.view(1, 1, img.size(0), img.size(1)), (SIZE, SIZE), mode="bilinear")[0, 0]

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img

        # Back to CPU + SAVE
        img = (img * 255).cpu().numpy().astype(np.uint8)

        cv2.imwrite(SAVE_FOLDER + f"{patient}_{image}.png", img)

    shutil.rmtree(J2K_FOLDER)



import dicomsdl

def dicomsdl_to_numpy_image(dicom, index=0):
    info = dicom.getPixelDataInfo()
    dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    outarr = np.empty(shape, dtype=dtype)
    dicom.copyFrameData(index, outarr)
    return outarr

def load_img_dicomsdl(f):
    return dicomsdl_to_numpy_image(dicomsdl.open(f))


def process(f, size=512, save_folder=""):
    patient = f.split('/')[-2]
    image = f.split('/')[-1][:-4]

    dicom = pydicom.dcmread(f)

    if dicom.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.4.90':  # ALREADY PROCESSED
        return

    try:
        img = load_img_dicomsdl(f)
    except:
        img = dicom.pixel_array

    img = (img - img.min()) / (img.max() - img.min())

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = 1 - img

    img = cv2.resize(img, (size, size))

    cv2.imwrite(save_folder + f"{patient}_{image}.png", (img * 255).astype(np.uint8))


_ = Parallel(n_jobs=2)(
    delayed(process)(img, size=SIZE, save_folder=SAVE_FOLDER)
    for img in tqdm(test_images)
)