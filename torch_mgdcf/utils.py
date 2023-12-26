import requests
from tqdm import tqdm
import os
import zipfile
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.sampler import BatchSampler

def download_file(url, download_path):

    if os.path.exists(download_path):
        print("File {} already exists".format(download_path))
        return
    
    print("Downloading file from {} to {}".format(url, download_path))

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(download_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")




def extract_zip(input_zip_path, output_path):

    with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_path)


def create_tensor_dataloader(tensor, batch_size=None, shuffle=False):
    dataset = TensorDataset(tensor)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(dataset, 
                        sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
                        collate_fn=lambda batchs: batchs[0][0]
                        )