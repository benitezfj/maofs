import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.utils import resample #Balance the dataset

def balance_train(features, label):
    train_df = pd.concat([features, label], axis=1)
    label_counts = train_df[label.name].value_counts()
    majority_label = label_counts.idxmax()
    balanced_train_df = pd.concat([train_df[train_df[label.name] != majority_label], train_df[train_df[label.name] == majority_label].sample(label_counts.min())])

    features = balanced_train_df.drop([label.name], axis=1)
    label = balanced_train_df[label.name]

    return features, label


def encode_labels(y_train, y_test, y_val):
    le = LabelEncoder().fit(y_train.values)
    y_train = pd.Series(le.transform(y_train.values))
    y_test = pd.Series(le.transform(y_test.values))
    y_val = pd.Series(le.transform(y_val.values))

    return y_train, y_test, y_val


def train_test_val_split(features, classes, balance=False, encoder=True, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    # _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Balancing the train dataset by removing the instance whose label is the majority
    if balance:
        X_train, y_train = balance_train(X_train, y_train)

    if encoder:
        y_train, y_test, y_val = encode_labels(y_train, y_test, y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val
