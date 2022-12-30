# https://www.kaggle.com/general/51898
pip install -r requirements.txt
mkdir -p ~/.kaggle/
cp kaggle.json  ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
# Download data
kaggle datasets download -d awsaf49/rsna-bcd-roi-1024x-png-dataset
# Unzip
unzip rsna-bcd-roi-1024x-png-dataset.zip -d 1024_data
# Run the pipeline
python train.py