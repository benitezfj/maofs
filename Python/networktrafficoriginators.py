import numpy as np

import os

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA

from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.algorithms.moo.nsga3 import comp_by_cv_then_random

from pymoo.core.problem import StarmapParallelization
from pymoo.util.ref_dirs import get_reference_directions

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from methods.mi_calculation import calculate_mutual_info
from methods.load_data import load_and_prepare_data
from methods.objective_function import FeatureSelectionManyProblem


def optimize_model(model_name, dataset_name, algorithm_name, iterator):
    STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

    features, classes = load_and_prepare_data(STORAGE_DIR,dataset_name)
    mutual_info_selected, features_subset = calculate_mutual_info(features, classes)

    if model_name == 'knn':
        model = KNeighborsClassifier()
    else:
        model = GaussianNB()

    feature_names = features_subset.columns
    feature_costs = np.ones(len(feature_names))

    ref_dirs = get_reference_directions("das-dennis", 4, n_partitions=7)
    ref_point = ref_dirs.max(axis=0)

    n_threads = 4
    pool = ThreadPoolExecutor(max_workers=n_threads)
    runner = StarmapParallelization(pool.starmap)

    problem = FeatureSelectionManyProblem(X=features_subset.values,
                                          y=classes,
                                          test_size=0.3,
                                          estimator=model,
                                          feature_names=feature_names,
                                          feature_costs=feature_costs,
                                          mutual_info=mutual_info_selected.values,
                                          elementwise_runner=runner)
    
    algorithm = None  # Initialize algorithm instance

    if algorithm_name == "RVEA":
      # RVEA Evaluation
      algorithm = RVEA(pop_size=120,
                    sampling=BinaryRandomSampling(),
                    selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                    crossover=TwoPointCrossover(),
                    mutation=BitflipMutation(),
                    ref_dirs=ref_dirs,
                    eliminate_duplicates=False)
    elif algorithm_name == "MOEAD":
      # MOEAD Evaluation
      algorithm = MOEAD(ref_dirs,
                      n_neighbors=15, 
                      prob_neighbor_mating=0.7, 
                      sampling=BinaryRandomSampling(), 
                      crossover=TwoPointCrossover(), 
                      mutation=BitflipMutation())
    elif algorithm_name == "NSGAIII":
      # NSGA-III Evaluation
      algorithm = NSGA3(pop_size=120,
                          sampling=BinaryRandomSampling(),
                          selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                          crossover=TwoPointCrossover(),
                          mutation=BitflipMutation(),
                          ref_dirs=ref_dirs,
                          eliminate_duplicates=False)
    elif algorithm_name == "NSGAII":
      # NSGAII Evaluation
      algorithm = NSGA2(pop_size=120,
                          sampling=BinaryRandomSampling(),
                          selection=TournamentSelection(func_comp=binary_tournament),
                          crossover=TwoPointCrossover(),
                          mutation=BitflipMutation(),
                          eliminate_duplicates=False)

    optimize_and_save(ref_dirs, dataset_name, algorithm, algorithm_name, model, model_name, problem, num_iterations)


model_names = ['gnb', 'knn']
dataset_aliases = ["linux_memory", "linux_disk", "linux_process", "network", "win7", "win10"]
algorithm_aliases = ["NSGA-III", "NSGA-II", "RVEA", "MOEA/D"]
num_iterations = 12

for model_name in model_names:
    for dataset_alias in dataset_aliases:
      for algorithm_alias in algorithm_aliases:
        algorithm_name = algorithm_alias
        dataset_name = dataset_alias
        optimize_model(model_name, dataset_name, algorithm_name, num_iterations)