import os
import numpy as np
import multiprocessing

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA

from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.algorithms.moo.nsga3 import comp_by_cv_then_random
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection

from pymoo.core.problem import StarmapParallelization
from pymoo.util.ref_dirs import get_reference_directions
# from multiprocessing.pool import ThreadPool

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from methods.splitting_data import balance_train_smote, balance_train_random, balance_train_adasyn, encode_labels  # ,train_test_val_split

from methods.su_calculation import calculate_symmetric_uncertainty
from methods.load_data import load_and_prepare_data
from methods.objective_function import FeatureSelectionManyProblem
from optimize_process.optimization_evaluation import optimize_and_save

def optimize_model(model_name, dataset_name, algorithm_name, n_outer_folds, n_inner_folds):
    # Load the dataset and separate features and classes
    # STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    STORAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
    features, classes = load_and_prepare_data(STORAGE_DIR, dataset_name)

    outer_cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

    for fold_idx, (outer_train_idx, test_idx) in enumerate(outer_cv.split(features, classes)):

        # Split outer train/test
        X_outer_train, X_test = (features.iloc[outer_train_idx], features.iloc[test_idx])
        y_outer_train, y_test = (classes.iloc[outer_train_idx], classes.iloc[test_idx])

        print(f"Outer Test Set Dimensions: X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Further split outer_train into inner_train and inner_val for inner CV
        inner_cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)

        for inner_fold_idx, (inner_train_idx, val_idx) in enumerate(inner_cv.split(X_outer_train, y_outer_train)):
            print(f"Outer Fold {fold_idx + 1}/{n_outer_folds}")
            print(f"Inner Fold {inner_fold_idx + 1}/{n_inner_folds}")

            # Split inner train/test
            # X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(features, classes, encoder=True, balance=False, random_state=42)
            X_train, X_val = (X_outer_train.iloc[inner_train_idx], X_outer_train.iloc[val_idx])
            y_train, y_val = (y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[val_idx])

            # Optional: Balance and encode the training data
            # X_train, y_train = balance_train_smote(X_train, y_train, 42)
            # X_train, y_train = balance_train_random(X_train, y_train, 42)
            X_train, y_train = balance_train_adasyn(X_train, y_train, 42)
            y_train, y_test, y_val = encode_labels(y_train, y_test, y_val)

            print(f"Inner Training Set Dimensions: X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"Inner Validation Set Dimensions: X_val: {X_val.shape}, y_val: {y_val.shape}")

            # Measure symmetric uncertainty and select features
            symmetric_info_selected = calculate_symmetric_uncertainty(X_train, y_train)  # ,features_subset
            X_train_subset = X_train[symmetric_info_selected.index]
            X_test_subset = X_test[symmetric_info_selected.index]
            X_val_subset = X_val[symmetric_info_selected.index]

            feature_names = X_train_subset.columns
            feature_costs = np.ones(len(feature_names))

            # mutual_info_selected, features_subset = calculate_mutual_info(X_train, y_train)

            if model_name == "knn":
                model = KNeighborsClassifier()
            elif model_name == "rf":
                model = RandomForestClassifier()
            elif model_name == "gnb":
                model = GaussianNB()

            # n_threads = 10
            # pool = ThreadPool(n_threads)
            # runner = StarmapParallelization(pool.starmap)
            # initialize the thread pool and create the runner
            n_proccess = 4
            pool = multiprocessing.Pool(n_proccess)
            runner = StarmapParallelization(pool.starmap)

            problem = FeatureSelectionManyProblem(X_train=X_train_subset.values,
						  X_test=X_val_subset.values,
						  # X_test=X_test_subset.values,
						  y_train=y_train,
						  y_test=y_val,
						  # y_test=y_test,
						  estimator=model,
						  feature_names=feature_names,
						  feature_costs=feature_costs,
						  mutual_info=symmetric_info_selected.values,
						  elementwise_runner=runner)

            ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=7)
            algorithm = None  # Initialize algorithm instance

            if algorithm_name == "RVEA":
                # RVEA Evaluation
                algorithm = RVEA(
                    pop_size=120,
                    sampling=BinaryRandomSampling(),
                    selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                    crossover=TwoPointCrossover(prob=0.8),
                    mutation=BitflipMutation(prob=0.1),
                    ref_dirs=ref_dirs,
                    eliminate_duplicates=False,
                )
            elif algorithm_name == "MOEAD":
                # MOEAD Evaluation
                algorithm = MOEAD(
                    ref_dirs,
                    n_neighbors=15,
                    prob_neighbor_mating=0.7,
                    sampling=BinaryRandomSampling(),
                    crossover=TwoPointCrossover(prob=0.8),
                    mutation=BitflipMutation(prob=0.1),
                )
            elif algorithm_name == "NSGA-III":
                # NSGA-III Evaluation
                algorithm = NSGA3(
                    pop_size=120,
                    sampling=BinaryRandomSampling(),
                    selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                    crossover=TwoPointCrossover(prob=0.8),
                    mutation=BitflipMutation(prob=0.1),
                    ref_dirs=ref_dirs,
                    eliminate_duplicates=False,
                )
            elif algorithm_name == "NSGA-II":
                # NSGAII Evaluation
                algorithm = NSGA2(
                    pop_size=120,
                    sampling=BinaryRandomSampling(),
                    selection=TournamentSelection(func_comp=binary_tournament),
                    crossover=TwoPointCrossover(prob=0.8),
                    mutation=BitflipMutation(prob=0.1),
                    eliminate_duplicates=False,
                )

            optimize_and_save(ref_dirs=ref_dirs,
			      dataset_name=dataset_name,
			      algorithm_instance=algorithm,
			      algorithm_name=algorithm_name,
			      model_instance=model,
			      model_name=model_name,
			      problem=problem,
			      train_features=X_train_subset.values,
			      train_classes=y_train,
			      # val_features=X_val_subset.values,
			      val_features=X_test_subset.values,
			      # val_classes=y_val,
			      val_classes=y_test,
			      filter_info=symmetric_info_selected.values,
			      outer_fold=fold_idx,
			      inner_fold=inner_fold_idx)