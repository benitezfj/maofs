import numpy as np
import autograd.numpy as anp

from pymoo.core.problem import ElementwiseProblem

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.base import clone


class FeatureSelectionManyProblem(ElementwiseProblem):
    def __init__(self, X, y, test_size, estimator, feature_names, feature_costs, mutual_info, scale_features=0.5, objectives=4, random_state=0, **kwargs):
        # self.y = y
        self.test_size = test_size
        self.estimator = estimator
        self.L = feature_names
        self.n_max = len(self.L)
        self.feature_costs = feature_costs
        self.mutual_info = mutual_info
        self.scale_features = scale_features
        self.objectives = objectives
        
        # If test size is not specify or it is 0, everything is took to test and train
        if self.test_size != 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=random_state)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = np.copy(X), np.copy(y), np.copy(X), np.copy(y)

        super().__init__(n_var=len(self.L), n_obj=objectives, elementwise_evaluation=True, **kwargs)

    def validation(self, x):
        # print(x)
        clf = clone(self.estimator)
        if all(not element for element in x):
            metrics = metrics1 = 0
            return metrics, metrics1
        else:
            clf.fit(self.X_train[:, x], self.y_train)
            y_pred = clf.predict(self.X_test[:, x])
            metrics = accuracy_score(self.y_test, y_pred)
            metrics1 = f1_score(self.y_test, y_pred, labels=list(set(self.y_train.values)), average='macro')
            return metrics, metrics1
        
    def _evaluate(self, x, out, *args, **kwargs):
        scores, scores1 =  self.validation(x)

        costs_selected = []
        feature_costs = np.array(self.feature_costs)
        costs_selected = feature_costs[np.argwhere(x==True)]
        cost_sum = sum(costs_selected)/sum(feature_costs)
        mutual_info = np.array(self.mutual_info)
        mutual_info_costs = sum(mutual_info[np.argwhere(x==True)])/sum(mutual_info)

        if cost_sum == 0:
            out["F"] = anp.column_stack(np.array([0, 0, 0, 0]))
        else:
            f1 = -1 * scores
            f2 = cost_sum
            f2 = 1 * f2[0]
            f3 = mutual_info_costs
            f3 = -1 * f3[0] #f3 = 1 * f3[0]
            f4 = -1 * scores1
            f1 = f1.item()
            f2 = f2.item()
            f3 = f3.item()
            f4 = f4.item()
            out["F"] = anp.column_stack(np.array([f1, f2, f3, f4]))    
        return out