import os
import pandas as pd

def load_and_prepare_data(storage, dataset_name):
    dataset_path = os.path.join(storage, dataset_name, f"{dataset_name}_normalize.csv")
    dataset = pd.read_csv(dataset_path, header=0, encoding='utf-8', skip_blank_lines=False, delimiter=r",")

    features = dataset.drop(['type'], axis=1)
    classes = dataset['type']

    return features, classes