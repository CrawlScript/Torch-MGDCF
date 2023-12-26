
from torch_mgdcf.utils import download_file, extract_zip
import os
import pickle
import numpy as np
import torch

def _read_edge_info(file_path):
    edge_dict = {}
    edges = []

    with open(file_path, "r", encoding="utf-8") as f:
        for l in f.readlines():
            if len(l) > 0:
                try:
                    l = l.strip('\n').split(' ')
                    items = []
                    uid = int(l[0])
                    for i in l[1:]:
                        i = int(i)
                        items.append(i)
                        edges.append([uid, i])
                    if uid not in edge_dict:
                        edge_dict[uid] = set(items)
                    else:
                        item_set = edge_dict[uid]
                        edge_dict[uid] = set(items).union(item_set)
                except Exception:
                    continue

    edges = np.array(edges)
    return edge_dict, edges

def _process(dataset_unzip_path):

    train_file = os.path.join(dataset_unzip_path, 'train.txt')
    test_file = os.path.join(dataset_unzip_path, 'test.txt')

    # print(train_file)
    # asdfasdf

    train_user_items_dict, train_user_item_edges = _read_edge_info(train_file)
    test_user_items_dict, test_user_item_edges = _read_edge_info(test_file)

    user_item_edges = np.concatenate([train_user_item_edges, test_user_item_edges], axis=0)
    index = np.arange(user_item_edges.shape[0])
    num_train_edges = train_user_item_edges.shape[0]
    train_index, test_index = index[:num_train_edges], index[num_train_edges:]
    num_users, num_items = user_item_edges.max(axis=0) + 1


    return num_users, num_items, user_item_edges, train_index, test_index, train_user_items_dict, test_user_items_dict





def load_dataset(dataset_name, data_root_path="./datasets", cache_name="cache.p"):
    """
    Load the DGL dataset.
    :param dataset_name: "yelp" | "gowalla" | "amazon-book"
    :param dataset_root_path:
    :return:
    """

    dataset_root_path = os.path.join(data_root_path, dataset_name)
    processed_root_path = os.path.join(dataset_root_path, "processed")
    cache_path = None if cache_name is None else os.path.join(processed_root_path, cache_name)
    raw_root_path = os.path.join(dataset_root_path, "raw")
    download_root_path = os.path.join(dataset_root_path, "download")
    download_file_name="{}.zip".format(dataset_name)
    download_file_path = os.path.join(download_root_path, download_file_name)
    download_url = "https://github.com/maenzhier/grecx_datasets/raw/main/{}/{}.zip".format(dataset_name.replace("light_gcn_", ""), dataset_name)
    dataset_unzip_path = os.path.join(raw_root_path, dataset_name)

    for path in [dataset_root_path, processed_root_path, raw_root_path, download_root_path]:
        if not os.path.exists(path):
            os.makedirs(path)



    if cache_path is not None and os.path.exists(cache_path):
        print("cache file exists: {}, read cache".format(cache_path))
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
        return dataset



    if not os.path.exists(download_file_path):
        download_file(download_url, download_file_path)
    

    if len(os.listdir(raw_root_path)) == 0:
        extract_zip(download_file_path, raw_root_path)

    processed = _process(dataset_unzip_path)

    if cache_path is not None:
        print("save processed data to cache: ", cache_path)
        with open(cache_path, "wb") as f:
            pickle.dump(processed, f)


    return processed