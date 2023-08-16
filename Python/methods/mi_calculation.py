import pandas as pd

from sklearn.feature_selection import mutual_info_classif


def calculate_mutual_info(features, classes, seed=42):
    np.random.seed(seed)
    mutual_info = mutual_info_classif(features, classes)
    np.random.seed(None)

    mutual_info_series = pd.Series(mutual_info, index=features.columns)
    mutual_info_selected = mutual_info_series[mutual_info_series > 0.0]
    features_subset = features[mutual_info_selected.index]

    return mutual_info_selected, features_subset