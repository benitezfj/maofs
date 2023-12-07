import pandas as pd
import numpy as np
from methods.symmetrical_uncertainty import su_measure

def calculate_symmetric_uncertainty(features, classes, seed=42):
    np.random.seed(seed)
    symmetric_uncer = su_measure(features.values.astype(float),classes.values)
    np.random.seed(None)

    symmetric_uncer_series = pd.Series(symmetric_uncer, index=features.columns)
    symmetric_uncer_selected = symmetric_uncer_series[symmetric_uncer_series > 0.0]
    # features_subset = features[symmetric_uncer_selected.index]

    return symmetric_uncer_selected #, features_subset