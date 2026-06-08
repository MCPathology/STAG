from huggingface_hub import login
import os
import zipfile
from huggingface_hub import snapshot_download
import datasets
import pandas as pd
from tqdm import tqdm

login(token=os.environ.get("HF_TOKEN"))

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")


data_path = "./data/hest1k_datasets/Liver"


def download_hest(patterns, local_dir):
    repo_id = 'MahmoodLab/hest'
    snapshot_download(repo_id=repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_dir)

    seg_dir = os.path.join(local_dir, 'cellvit_seg')
    if os.path.exists(seg_dir):
        print('Unzipping cell vit segmentation...')
        for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
            path_zip = os.path.join(seg_dir, filename)
                        
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(seg_dir)


data_path = "./data/hest1k_datasets/Bone"

meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_1_0.csv")

meta_df = meta_df[meta_df['organ'] == 'Bone']

ids_to_query = meta_df['id'].values

list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
print(list_patterns)
download_hest(list_patterns, data_path) 
