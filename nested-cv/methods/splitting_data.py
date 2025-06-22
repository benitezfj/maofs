import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # , StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from collections import Counter

# from sklearn.utils import resample #Balance the dataset


def balance_train(features, label):
    train_df = pd.concat([features, label], axis=1)
    label_counts = train_df[label.name].value_counts()
    majority_label = label_counts.idxmax()
    balanced_train_df = pd.concat(
        [
            train_df[train_df[label.name] != majority_label],
            train_df[train_df[label.name] == majority_label].sample(label_counts.min()),
        ]
    )

    features = balanced_train_df.drop([label.name], axis=1)
    label = balanced_train_df[label.name]

    return features, label


def balance_train_smote(features, label, random_state):
    # Get the class distribution
    label_counts = Counter(label)

    num_classes = len(label_counts)

    # If binary classification, oversample only the minority class
    if num_classes == 2:
        # Identify the minority class
        minority_class = min(label_counts, key=label_counts.get)

        # Oversample the minority class to match the majority class size
        sampling_strategy_smote = {
            minority_class: label_counts[max(label_counts, key=label_counts.get)]
        }

        # Adjust k_neighbors for binary case
        smote_k_neighbors = min(5, label_counts[minority_class] - 1)

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),  # Replace with StandardScaler() if needed
                (
                    "oversample",
                    SMOTE(
                        sampling_strategy=sampling_strategy_smote,
                        k_neighbors=smote_k_neighbors,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        # Apply the pipeline to balance the dataset
        features_resampled, label_resampled = pipeline.fit_resample(features, label)

    else:  # Multiclass case
        # Identify the majority class
        majority_class = max(label_counts, key=label_counts.get)

        # Exclude the majority class to find the largest minority class size
        minority_classes = {
            k: v for k, v in label_counts.items() if k != majority_class
        }
        target_size = max(minority_classes.values())  # Largest minority class size

        # Define the sampling strategies
        sampling_strategy_smote = {
            class_label: target_size
            for class_label in label_counts
            if label_counts[class_label] < target_size
        }
        sampling_strategy_undersample = {majority_class: target_size}

        # Handle small classes by adjusting k_neighbors
        smote_k_neighbors = min(
            5, min(label_counts.values()) - 1
        )  # Ensure no error in SMOTE

        # Create the balancing pipeline
        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),  # Replace with StandardScaler() if needed
                (
                    "oversample",
                    SMOTE(
                        sampling_strategy=sampling_strategy_smote,
                        k_neighbors=smote_k_neighbors,
                        random_state=random_state,
                    ),
                ),
                (
                    "undersample",
                    RandomUnderSampler(
                        sampling_strategy=sampling_strategy_undersample,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        # Apply the pipeline to balance the dataset
        features_resampled, label_resampled = pipeline.fit_resample(features, label)

    # Convert the resampled features and label back to DataFrame and Series respectively
    features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
    label_resampled = pd.Series(label_resampled, name=label.name)

    # Display the class distribution after balancing
    print("Original class distribution:", label_counts)
    print("Balanced class distribution:", Counter(label_resampled))

    return features_resampled, label_resampled


def balance_train_oversample(features, label, random_state):
    # Get the class distribution
    label_counts = Counter(label)
    num_classes = len(label_counts)

    if num_classes == 2:  # Binary case
        minority_class = min(label_counts, key=label_counts.get)
        sampling_strategy = {
            minority_class: label_counts[max(label_counts, key=label_counts.get)]
        }

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    RandomOverSampler(
                        sampling_strategy=sampling_strategy, random_state=random_state
                    ),
                ),
            ]
        )
    else:  # Multiclass case
        majority_class = max(label_counts, key=label_counts.get)
        minority_classes = {
            k: v for k, v in label_counts.items() if k != majority_class
        }
        target_size = max(minority_classes.values())

        sampling_strategy_oversample = {
            class_label: target_size
            for class_label in label_counts
            if label_counts[class_label] < target_size
        }
        sampling_strategy_undersample = {majority_class: target_size}

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    RandomOverSampler(
                        sampling_strategy=sampling_strategy_oversample,
                        random_state=random_state,
                    ),
                ),
                (
                    "undersample",
                    RandomUnderSampler(
                        sampling_strategy=sampling_strategy_undersample,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    features_resampled, label_resampled = pipeline.fit_resample(features, label)
    features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
    label_resampled = pd.Series(label_resampled, name=label.name)

    print("Original class distribution:", label_counts)
    print("Balanced class distribution:", Counter(label_resampled))
    return features_resampled, label_resampled


def balance_train_adasyn(features, label, random_state):
    # Get the class distribution
    label_counts = Counter(label)
    num_classes = len(label_counts)

    if num_classes == 2:  # Binary case
        minority_class = min(label_counts, key=label_counts.get)
        sampling_strategy = {
            minority_class: label_counts[max(label_counts, key=label_counts.get)]
        }

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    ADASYN(
                        sampling_strategy=sampling_strategy, random_state=random_state
                    ),
                ),
            ]
        )
    else:  # Multiclass case
        majority_class = max(label_counts, key=label_counts.get)
        minority_classes = {
            k: v for k, v in label_counts.items() if k != majority_class
        }
        target_size = max(minority_classes.values())

        sampling_strategy_oversample = {
            class_label: target_size
            for class_label in label_counts
            if label_counts[class_label] < target_size
        }
        sampling_strategy_undersample = {majority_class: target_size}

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    ADASYN(
                        sampling_strategy=sampling_strategy_oversample,
                        random_state=random_state,
                    ),
                ),
                (
                    "undersample",
                    RandomUnderSampler(
                        sampling_strategy=sampling_strategy_undersample,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    features_resampled, label_resampled = pipeline.fit_resample(features, label)
    features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
    label_resampled = pd.Series(label_resampled, name=label.name)

    print("Original class distribution:", label_counts)
    print("Balanced class distribution:", Counter(label_resampled))
    return features_resampled, label_resampled


def balance_train_undersample(features, label, random_state):
    # Get the class distribution
    label_counts = Counter(label)
    num_classes = len(label_counts)

    if num_classes == 2:  # Binary case
        minority_class = min(label_counts, key=label_counts.get)
        sampling_strategy = {
            minority_class: label_counts[max(label_counts, key=label_counts.get)]
        }

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    RandomOverSampler(
                        sampling_strategy=sampling_strategy, random_state=random_state
                    ),
                ),
                #           ("undersample", AllKNN(n_neighbors=3)),
                #           ("undersample", ClusterCentroids(random_state=random_state)),
                #           ("undersample", NearMiss(version=1)),
            ]
        )
    else:  # Multiclass case
        majority_class = max(label_counts, key=label_counts.get)
        minority_classes = {
            k: v for k, v in label_counts.items() if k != majority_class
        }
        target_size = max(minority_classes.values())

        sampling_strategy_oversample = {
            class_label: target_size
            for class_label in label_counts
            if label_counts[class_label] < target_size
        }
        sampling_strategy_undersample = {majority_class: target_size}

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    RandomOverSampler(
                        sampling_strategy=sampling_strategy_oversample,
                        random_state=random_state,
                    ),
                ),
                # ("oversample", ADASYN(sampling_strategy=sampling_strategy_oversample, random_state=random_state)),
                ("undersample", AllKNN(n_neighbors=3)),
                # ("undersample", RandomUnderSampler(sampling_strategy=sampling_strategy_undersample, random_state=random_state)),
                # ("undersample", ClusterCentroids(random_state=random_state)),
                # ("undersample", NearMiss(version=1)),
            ]
        )

    features_resampled, label_resampled = pipeline.fit_resample(features, label)
    features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
    label_resampled = pd.Series(label_resampled, name=label.name)

    print("Original class distribution:", label_counts)
    print("Balanced class distribution:", Counter(label_resampled))
    return features_resampled, label_resampled


def balance_train_nearmiss(features, label, random_state):
    # Get the class distribution
    label_counts = Counter(label)
    num_classes = len(label_counts)

    if num_classes == 2:  # Binary case
        minority_class = min(label_counts, key=label_counts.get)
        sampling_strategy = {
            minority_class: label_counts[max(label_counts, key=label_counts.get)]
        }

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    RandomOverSampler(
                        sampling_strategy=sampling_strategy, random_state=random_state
                    ),
                ),
            ]
        )
    else:  # Multiclass case
        majority_class = max(label_counts, key=label_counts.get)
        minority_classes = {
            k: v for k, v in label_counts.items() if k != majority_class
        }
        target_size = max(minority_classes.values())

        sampling_strategy_oversample = {
            class_label: target_size
            for class_label in label_counts
            if label_counts[class_label] < target_size and class_label != majority_class
        }

        sampling_strategy_undersample = {majority_class: target_size}

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "oversample",
                    RandomOverSampler(
                        sampling_strategy=sampling_strategy_oversample,
                        random_state=random_state,
                    ),
                ),
                (
                    "undersample",
                    NearMiss(
                        sampling_strategy=sampling_strategy_undersample, version=1
                    ),
                ),
            ]
        )

    features_resampled, label_resampled = pipeline.fit_resample(features, label)
    features_resampled = pd.DataFrame(features_resampled, columns=features.columns)
    label_resampled = pd.Series(label_resampled, name=label.name)

    print("Original class distribution:", label_counts)
    print("Balanced class distribution:", Counter(label_resampled))
    return features_resampled, label_resampled


def encode_labels(y_train, y_test, y_val):
    le = LabelEncoder().fit(y_train.values)
    y_train = pd.Series(le.transform(y_train.values))
    y_test = pd.Series(le.transform(y_test.values))
    y_val = pd.Series(le.transform(y_val.values))

    return y_train, y_test, y_val


def train_test_val_split(
    features, classes, balance=False, encoder=True, random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        features, classes, stratify=classes, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, test_size=0.2, random_state=random_state
    )
    # _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Balancing the train dataset by removing the instance whose label is the majority
    if balance:
        # print("Dataset class distribution:", Counter(y_train))
        X_train, y_train = balance_train(X_train, y_train, random_state)

    if encoder:
        y_train, y_test, y_val = encode_labels(y_train, y_test, y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val
