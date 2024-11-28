import os
import pandas as pd
from methods.encode_normalizate import coding_normalization


def load_and_prepare_data(storage, dataset_name):
    dataset_path = os.path.join(storage, f"{dataset_name}_normalize.csv")
    dataset = pd.read_csv(dataset_path, header=0, encoding='utf-8', skip_blank_lines=False, delimiter=r",")
    #dataset = coding_normalization(dataset)
    features = dataset.drop(['type', 'label'], axis=1)
    classes = dataset['label']
    return features, classes



# def load_and_prepare_data(storage, dataset_name):
#     dataset_path = os.path.join(storage, dataset_name, f"{dataset_name}_normalize.csv")
#     dataset = pd.read_csv(dataset_path, header=0, encoding='utf-8', skip_blank_lines=False, delimiter=r",")
#     features = dataset.drop(['type', 'label'], axis=1)
#     if dataset_name in ["linux_memory", "linux_disk", "win7"]:
#         features = dataset.drop(['type'], axis=1)
#     else:
#         features = dataset.drop(['type', 'label'], axis=1)
#     classes = dataset['label']
#     classes = dataset['type']
#     return features, classes