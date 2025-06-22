import pandas as pd

from sklearn.preprocessing import MinMaxScaler  # , StandardScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

def balance_smote(features, label, random_state):
    label_counts = Counter(label)
    num_classes = len(label_counts)

    min_class_size = min(label_counts.values())
    smote_k_neighbors = max(1, min(5, min_class_size - 1))

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
    label_name = getattr(label, "name", "label")  # Use 'label' if .name does not exist
    label_resampled = pd.Series(label_resampled, name=label_name)


    # Display the class distribution after balancing
    print("Original class distribution:", label_counts)
    print("Balanced class distribution with Smote Method:", Counter(label_resampled))

    return features_resampled, label_resampled