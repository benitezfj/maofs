import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # , StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.utils import resample #Balance the dataset

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

def train_test_val_split(features, classes, encoder, balance, random_state):
    X_train, X_test, y_train, y_test = train_test_split(features, classes, stratify=classes, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state)
    # _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
    print("Dataset class distribution:", Counter(y_train))
    
    def balance_train(features, label):
        # Exclude the "normal" class from determining the largest class size
        label_counts = Counter(label)
        non_normal_classes = {k: v for k, v in label_counts.items() if k != "normal"}
        target_size = max(non_normal_classes.values())  # Find the largest non-normal class size

        # Define the sampling strategies
        sampling_strategy_smote = {
            class_label: target_size
            for class_label in label_counts
            if label_counts[class_label] < target_size and class_label != "normal"
        }
        sampling_strategy_undersample = {"normal": target_size}

        # Handle small classes by adjusting k_neighbors
        smote_k_neighbors = min(5, min(label_counts.values()) - 1)  # Ensure no error in SMOTE

        pipeline = Pipeline([
                ("scaler", MinMaxScaler()),  # Replace with StandardScaler() if needed
                ("oversample", SMOTE(sampling_strategy=sampling_strategy_smote, k_neighbors=smote_k_neighbors, random_state=random_state)),
                ("undersample", RandomUnderSampler(sampling_strategy=sampling_strategy_undersample, random_state=random_state)),
        ])

        features_resampled, label_resampled = pipeline.fit_resample(features, label)

        # Convert the resampled features and label back to DataFrame and Series respectively
        features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
        label_resampled = pd.Series(label_resampled, name=label.name)
        print("Balanced dataset class distribution:", Counter(label_resampled))

        return features_resampled, label_resampled

    # def balance_train(features, label):
    #     train_df = pd.concat([features, label], axis=1)

    #     label_counts = train_df[label.name].value_counts()
    #     # label_counts = train_df["label"].value_counts()
    #     majority_label = label_counts.idxmax()

    #     # balanced_train_df = train_df[train_df['label'] != majority_label].append(train_df[train_df['label'] == majority_label].sample(label_counts.min()))

    #     balanced_train_df = pd.concat(
    #         [
    #             train_df[train_df[label.name] != majority_label],
    #             train_df[train_df[label.name] == majority_label].sample(
    #                 label_counts.min()
    #             ),
    #         ]
    #     )
    #     # balanced_train_df = pd.concat([train_df[train_df["label"] != majority_label],train_df[train_df["label"] == majority_label].sample(label_counts.min())])

    #     features = balanced_train_df.drop([label.name], axis=1)
    #     # features = balanced_train_df.drop(["label"], axis=1)

    #     label = balanced_train_df[label.name]
    #     # label = balanced_train_df["label"]
    #     return features, label

    # Balancing the train dataset by removing the instance whose label is the majority
    if balance:
        X_train, y_train = balance_train(X_train, y_train)

    # y_train_majority = y_train[y_train.values == y_train.value_counts().idxmax()]
    # y_train_minority = y_train[y_train.values == y_train.value_counts().idxmin()]
    # y_train_majority_downsampled = resample(y_train_majority,
    #                                         replace=False,
    #                                         n_samples=len(y_train_minority),
    #                                         random_state=123)
    # y_train_balanced = pd.concat([y_train_minority, y_train_majority_downsampled])
    # X_train.iloc[y_train_balanced.index]
    # np.bincount(y_train_balanced)

    if encoder:
        le = LabelEncoder().fit(y_train.values)
        y_train = pd.Series(le.transform(y_train.values))
        y_test = pd.Series(le.transform(y_test.values))
        y_val = pd.Series(le.transform(y_val.values))

    return X_train, X_test, X_val, y_train, y_test, y_val