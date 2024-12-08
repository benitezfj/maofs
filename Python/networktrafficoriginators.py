import os
import numpy as np

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
from multiprocessing.pool import ThreadPool

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from methods.splitting_data import train_test_val_split

from methods.su_calculation import calculate_symmetric_uncertainty
from methods.load_data import load_and_prepare_data
from methods.objective_function import FeatureSelectionManyProblem
from optimize_process.optimization_evaluation import optimize_and_save

def optimize_model(model_name, dataset_name, algorithm_name):
    STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

    features, classes = load_and_prepare_data(STORAGE_DIR, dataset_name)

    X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(features, classes, encoder=True, balance=True, random_state=42)

    #  mutual_info_selected, features_subset = calculate_mutual_info(X_train, y_train)
    symmetric_info_selected = calculate_symmetric_uncertainty(X_train, y_train)  # ,features_subset

    if model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "rf":
        model = RandomForestClassifier()
    elif model_name == "gnb":
        model = GaussianNB()

    X_train_subset = X_train[symmetric_info_selected.index]
    X_test_subset = X_test[symmetric_info_selected.index]
    X_val_subset = X_val[symmetric_info_selected.index]
    feature_names = X_train_subset.columns
    feature_costs = np.ones(len(feature_names))

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=7)

    n_threads = 10
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = FeatureSelectionManyProblem(X_train=X_train_subset.values,
					  X_test=X_test_subset.values,
					  y_train=y_train,
					  y_test=y_test,
					  estimator=model,
					  feature_names=feature_names,
					  feature_costs=feature_costs,
					  mutual_info=symmetric_info_selected.values,
					  elementwise_runner=runner)

    # problem = FeatureSelectionManyProblem(X=features_subset.values,
    #                                       y=classes,
    #                                       test_size=0.3,
    #                                       estimator=model,
    #                                       feature_names=feature_names,
    #                                       feature_costs=feature_costs,
    #                                       mutual_info=mutual_info_selected.values,
    #                                       elementwise_runner=runner)

    algorithm = None  # Initialize algorithm instance

    if algorithm_name == "RVEA":
        # RVEA Evaluation
        algorithm = RVEA(pop_size=120,
            sampling=BinaryRandomSampling(),
            selection=TournamentSelection(func_comp=comp_by_cv_then_random),
            crossover=TwoPointCrossover(prob=0.8),
            mutation=BitflipMutation(prob=0.1),
            ref_dirs=ref_dirs,
            eliminate_duplicates=False)
    elif algorithm_name == "MOEAD":
        # MOEAD Evaluation
        algorithm = MOEAD(ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(prob=0.8),
            mutation=BitflipMutation(prob=0.1))
    elif algorithm_name == "NSGA-III":
        # NSGA-III Evaluation
        algorithm = NSGA3(pop_size=120,
            sampling=BinaryRandomSampling(),
            selection=TournamentSelection(func_comp=comp_by_cv_then_random),
            crossover=TwoPointCrossover(prob=0.8),
            mutation=BitflipMutation(prob=0.1),
            ref_dirs=ref_dirs,
            eliminate_duplicates=False)
    elif algorithm_name == "NSGA-II":
        # NSGAII Evaluation
        algorithm = NSGA2(pop_size=120,
            sampling=BinaryRandomSampling(),
            selection=TournamentSelection(func_comp=binary_tournament),
            crossover=TwoPointCrossover(prob=0.8),
            mutation=BitflipMutation(prob=0.1),
            eliminate_duplicates=False)

    optimize_and_save(ref_dirs=ref_dirs,
        dataset_name=dataset_name,
        algorithm_instance=algorithm,
        algorithm_name=algorithm_name,
        model_instance=model,
        model_name=model_name,
        problem=problem,
        train_features=X_train_subset.values,
        train_classes=y_train,
        val_features=X_val_subset.values,
        val_classes=y_val,
        filter_info=symmetric_info_selected.values)
