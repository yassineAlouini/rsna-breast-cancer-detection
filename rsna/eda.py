from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

# Make Stratified K-Fold augmented dataset and save it.

sgkf = StratifiedGroupKFold(
    n_splits=4, shuffle=True, random_state=42
)

df = pd.read_csv("/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/train.csv")
patient_id_any_cancer = df.groupby('patient_id').cancer.max().reset_index()

# Need this to map label strings to ids for the dataset


for fold, (train_idx, val_idx) in enumerate(sgkf.split(patient_id_any_cancer.patient_id, patient_id_any_cancer.cancer)):
    df.loc[val_idx, "fold"] = fold

df.to_csv("/home/yassinealouini/Documents/Kaggle/rsna-breast-cancer-detection/1024_data/sgkf_train.csv")