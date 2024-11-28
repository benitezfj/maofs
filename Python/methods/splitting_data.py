import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.utils import resample #Balance the dataset


def train_test_val_split(features, classes, encoder):
    X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.3, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    def balance_train(features, label):
        train_df = pd.concat([features, label], axis=1)
        label_counts = train_df['label'].value_counts()
        majority_label = label_counts.idxmax()

        #balanced_train_df = train_df[train_df['label'] != majority_label].append(train_df[train_df['label'] == majority_label].sample(label_counts.min()))

        balanced_train_df = pd.concat([
            train_df[train_df['label'] != majority_label], 
            train_df[train_df['label'] == majority_label].sample(label_counts.min())
        ])

        features = balanced_train_df.drop(['label'], axis=1)
        label = balanced_train_df['label']
        return features, label
    
    #Balancing the train dataset by removing the instance whose label is the majority
    X_train, y_train = balance_train(X_train, y_train)
    
    # y_train_majority = y_train[y_train.values == y_train.value_counts().idxmax()]
    # y_train_minority = y_train[y_train.values == y_train.value_counts().idxmin()]
    # y_train_majority_downsampled = resample(y_train_majority,
    #                                         replace=False,
    #                                         n_samples=len(y_train_minority),
    #                                         random_state=123)
    # y_train_balanced = pd.concat([y_train_minority, y_train_majority_downsampled])    
    # X_train.iloc[y_train_balanced.index]
    #np.bincount(y_train_balanced)

    if encoder:
        le = LabelEncoder().fit(y_train.values)
        y_train = pd.Series(le.transform(y_train.values))
        y_test = pd.Series(le.transform(y_test.values))
        y_val = pd.Series(le.transform(y_val.values))

    return X_train, X_test, X_val, y_train, y_test, y_val