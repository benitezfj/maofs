import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # , StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, AllKNN

from collections import Counter

# from sklearn.utils import resample #Balance the dataset


def balance_train(features, label, random_state):
    """
    Balances the training dataset by undersampling the majority class.

    Parameters:
    features (pd.DataFrame): The feature set.
    label (pd.Series): The label set.
    random_state (int, optional): Random state for reproducibility.

    Returns:
    pd.DataFrame, pd.Series: Balanced feature set and label set.
    """
    train_df = pd.concat([features, label], axis=1)
    label_counts = train_df[label.name].value_counts()
    majority_label = label_counts.idxmax()

    # Separate majority and minority classes
    # balanced_train_df = pd.concat(
    #     [
    #         train_df[train_df[label.name] != majority_label],
    #         train_df[train_df[label.name] == majority_label].sample(label_counts.min(), random_state=random_state),
    #     ]
    # )
    minority_class_df = train_df[train_df[label.name] != majority_label]
    majority_class_df = train_df[train_df[label.name] == majority_label]

    # Sample from the majority class
    majority_class_sample = majority_class_df.sample(
        label_counts.min(), random_state=random_state
    )

    # Combine minority class and sampled majority class
    balanced_train_df = pd.concat([minority_class_df, majority_class_sample])

    # Shuffle the balanced dataframe
    balanced_train_df = balanced_train_df.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    features = balanced_train_df.drop([label.name], axis=1)
    label = balanced_train_df[label.name]

    return features, label


def balance_train_smote(features, label, random_state):
    label_counts = Counter(label)
    num_classes = len(label_counts)

    # Determine k_neighbors safely
    min_class_size = min(label_counts.values())
    smote_k_neighbors = max(1, min(5, min_class_size - 1))  # Avoid k > samples

    # If binary classification, oversample only the minority class
    if num_classes == 2:
        # Identify the minority class
        minority_class = min(label_counts, key=label_counts.get)

        # Oversample the minority class to match the majority class size
        sampling_strategy_smote = {
            minority_class: label_counts[max(label_counts, key=label_counts.get)]
        }

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
    print("Balanced class distribution with Smote Method:", Counter(label_resampled))

    return features_resampled, label_resampled


def balance_train_oversample(features, label, random_state):
    label_counts = Counter(label)
    num_classes = len(label_counts)

    if num_classes == 2:  # Binary classification
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
            cls: count for cls, count in label_counts.items() if cls != majority_class
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
    print(
        "Balanced class distribution with Oversample Method:", Counter(label_resampled)
    )
    return features_resampled, label_resampled


def balance_train_adasyn(features, label, random_state):
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
            cls: count for cls, count in label_counts.items() if cls != majority_class
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
    print("Balanced class distribution with Adasyn Method:", Counter(label_resampled))

    return features_resampled, label_resampled


def balance_train_undersample(features, label, random_state, method="random"):
    if method not in ["random", "nearmiss", "allknn"]:
        raise ValueError(
            f"Invalid method: {method}. Choose from 'random', 'nearmiss', or 'allknn'."
        )

    label_counts = Counter(label)
    num_classes = len(label_counts)
    majority_class = max(label_counts, key=label_counts.get)

    # Adjust for multiclass undersampling
    if num_classes > 2 and method in ["nearmiss", "allknn"]:
        raise ValueError(
            f"Method {method} does not support multiclass undersampling. Use 'random' instead."
        )

    if num_classes > 2:
        sampling_strategy = {cls: min(label_counts.values()) for cls in label_counts}
    else:
        sampling_strategy = {majority_class: min(label_counts.values())}

        undersampling_methods = {
            "random": RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            ),
            "nearmiss": NearMiss(sampling_strategy=sampling_strategy, version=1),
            "allknn": AllKNN(n_neighbors=3),
        }

    pipeline = Pipeline(
        [("scaler", MinMaxScaler()), ("undersample", undersampling_methods[method])]
    )

    features_resampled, label_resampled = pipeline.fit_resample(features, label)

    # Ensure columns match after undersampling
    features_resampled = pd.DataFrame(
        features_resampled, columns=features.columns[: features_resampled.shape[1]]
    )
    label_resampled = pd.Series(label_resampled, name=label.name)

    print("Original class distribution:", label_counts)
    print(
        "Balanced class distribution with Undersample method:", Counter(label_resampled)
    )

    return features_resampled, label_resampled


def balance_train_nearmiss(features, label, random_state):
    label_counts = Counter(label)
    num_classes = len(label_counts)

    if num_classes == 2:  # Binary case
        # minority_class = min(label_counts, key=label_counts.get)
        # sampling_strategy = {minority_class: label_counts[max(label_counts, key=label_counts.get)]}
        majority_class = max(label_counts, key=label_counts.get)
        sampling_strategy = {
            majority_class: label_counts[min(label_counts, key=label_counts.get)]
        }

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                # ("oversample", RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)),
                (
                    "undersample",
                    NearMiss(sampling_strategy=sampling_strategy, version=1),
                ),
            ]
        )
    else:  # Multiclass case
        majority_class = max(label_counts, key=label_counts.get)
        minority_classes = {
            cls: count for cls, count in label_counts.items() if cls != majority_class
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
    print("Balanced class distribution with Nearmiss:", Counter(label_resampled))
    return features_resampled, label_resampled


# def encode_labels(y_train, y_test, y_val):
#     le = LabelEncoder().fit(y_train.values)
#     y_train = pd.Series(le.transform(y_train.values))
#     y_test = pd.Series(le.transform(y_test.values))
#     y_val = pd.Series(le.transform(y_val.values))

#     return y_train, y_test, y_val


def encode_labels(y_train, y_test, y_val):
    le = LabelEncoder().fit(y_train)
    return map(lambda y: pd.Series(le.transform(y)), (y_train, y_test, y_val))


def train_test_val_split(features, classes, encoder, balance, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        features, classes, stratify=classes, test_size=0.2, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    # _, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)

    # Balancing the train dataset by removing the instance whose label is the majority
    if balance:
        # print("Dataset class distribution:", Counter(y_train))
        X_train, y_train = balance_train(X_train, y_train, random_state)

    if encoder:
        y_train, y_test, y_val = encode_labels(y_train, y_test, y_val)

    return X_train, X_test, X_val, y_train, y_test, y_val
