import numpy as np
import autograd.numpy as anp

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.base import clone


def featureSelectionMany(x, X_train, y_train, X_test, y_test, mutual_info, estimator):
    feature_costs = np.ones(X_train.shape[1])

    def validation(x, X_train, y_train, X_test, y_test, estimator):
        clf = clone(estimator)
        if all(not element for element in x):
            metrics = metrics1 = metrics2 = (
                0  # Return 0 for all metrics when no features are selected
            )

        else:
            clf.fit(X_train[:, x], y_train)
            y_pred = clf.predict(X_test[:, x])
            metrics = recall_score(y_test, y_pred, average="macro", zero_division=1)
            metrics1 = f1_score(
                y_test,
                y_pred,
                labels=list(set(y_train)),
                average="macro",
                zero_division=1,
            )
            metrics2 = accuracy_score(
                y_test, y_pred
            )  # Compute accuracy as the third metric
            # metrics1 = f1_score(y_test, y_pred, labels=list(set(y_train.values)), average='macro')

        return metrics, metrics1, metrics2

    scores, scores1, scores2 = validation(
        x, X_train, y_train, X_test, y_test, estimator
    )

    costs_selected = []
    feature_costs = np.array(feature_costs)
    costs_selected = feature_costs[np.argwhere(x == True)]
    cost_sum = sum(costs_selected) / sum(feature_costs)
    mutual_info = np.array(mutual_info)
    mutual_info_costs = sum(mutual_info[np.argwhere(x == True)]) / sum(mutual_info)

    if cost_sum == 0:
        out = anp.column_stack(np.array([0, 0, 0, 0, 0]))
    else:
        f1 = -1 * scores
        f2 = cost_sum
        f2 = 1 * f2[0]
        f3 = mutual_info_costs
        f3 = -1 * f3[0]
        f4 = -1 * scores1
        f5 = -1 * scores2
        f1 = f1.item()  # Specificity or Accuracy
        f2 = f2.item()  # Number of features selected
        f3 = f3.item()  # Symmetrical Uncentraintly
        f4 = f4.item()  # Macro F1 Score
        f5 = f5.item()  # Accuracy
        out = anp.column_stack(np.array([f1, f2, f3, f4, f5]))

    return out
