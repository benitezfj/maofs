import time
from networktrafficoriginators import optimize_model

model_names = ["knn"]
dataset_aliases = ["alt"]
algorithm_aliases = ["MOEAD"]
# model_names = ["gnb", "rf", "knn"]
# dataset_aliases = ["alt", "bcc", "bcg", "bco", "cln", "cns", "cro", "dis", "lng", "lym"]
# algorithm_aliases = ["NSGA-II", "NSGA-III", "RVEA", "MOEAD"]
# num_iterations = 20

if __name__ == "__main__":
    for model_name in model_names:
        for dataset_alias in dataset_aliases:
            for algorithm_alias in algorithm_aliases:
                start_time = time.time()
                optimize_model(model_name, dataset_alias, algorithm_alias, 5, 3)
                end_time = time.time()
                execution_time = end_time - start_time
                print(
                    f"Execution time for {model_name}, {dataset_alias}, {algorithm_alias}: {execution_time:.4f} seconds"
                )
