import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import dicomsdl
    
from pathlib import Path
import multiprocessing as mp
import cv2

# TODO: Use the trained models...

def dicom_file_to_ary(path):
    dcm_file = dicomsdl.open(str(path))
    data = dcm_file.pixelData()

    data = (data - data.min()) / (data.max() - data.min())

    if dcm_file.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        data = 1 - data

    data = cv2.resize(data, RESIZE_TO)
    data = (data * 255).astype(np.uint8)
    return data

directories = list(Path(TEST_DICOM_DIR).iterdir())

def process_directory(directory_path):
    parent_directory = str(directory_path).split('/')[-1]
    !mkdir -p test_resized_{RESIZE_TO[0]}/{parent_directory}
    for image_path in directory_path.iterdir():
        processed_ary = dicom_file_to_ary(image_path)
        cv2.imwrite(
            f'test_resized_{RESIZE_TO[0]}/{parent_directory}/{image_path.stem}.png',
            processed_ary
        )

with mp.Pool(mp.cpu_count()) as p:
    p.map(process_directory, directories)

# %%
%%time

preds_all = []

test_dl = learn.dls.test_dl(get_image_files(f'test_resized_{RESIZE_TO[0]}'))
for SPLIT in range(NUM_SPLITS):
    learn.load(f'{MODEL_PATH}/{SPLIT}')
    preds, _ = learn.get_preds(dl=test_dl)
    preds_all.append(preds)

# %%
preds = torch.zeros_like(preds_all[0])
for pred in preds_all:
    preds += pred

preds /= NUM_SPLITS


preds = optimize_preds(preds, thresh=threshold)
image_ids = [path.stem for path in test_dl.items]

image_id2pred = defaultdict(lambda: 0)
for image_id, pred in zip(image_ids, preds[:, 1]):
    image_id2pred[int(image_id)] = pred.item()

# %% [markdown]
# <a id="section-three"></a>
# # Making a submission

# %%
test_csv = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')

prediction_ids = []
preds = []

for _, row in test_csv.iterrows():
    prediction_ids.append(row.prediction_id)
    preds.append(image_id2pred[row.image_id])

submission = pd.DataFrame(data={'prediction_id': prediction_ids, 'cancer': preds}).groupby('prediction_id').max().reset_index()
submission.head()

# %%
submission.to_csv('submission.csv', index=False)

# %% [markdown]
# And that's it! Thank you very much for reading! 🙂
# 
# **If you enjoyed the notebook, please upvote! 🙏 Thank you, appreciate your support!**
# 
# Happy Kaggling 🥳
# 