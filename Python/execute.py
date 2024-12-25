import time
from networktrafficoriginators import optimize_model

model_names = ["knn"]
dataset_aliases = ["win7"]
algorithm_aliases = ["RVEA"]
# model_names = ["gnb", "rf", "knn"]
# dataset_aliases = ["linux_memory", "linux_disk", "linux_process", "network", "win7", "win10"]
# algorithm_aliases = ["NSGA-II", "NSGA-III", "RVEA", "MOEAD"]
# num_iterations = 20

if __name__ == "__main__":
    for model_name in model_names:
        for dataset_alias in dataset_aliases:
            for algorithm_alias in algorithm_aliases:
                start_time = time.time()
                optimize_model(model_name, dataset_alias, algorithm_alias)
                end_time = time.time()
                execution_time = end_time - start_time
                print(f"Execution time for {model_name}, {dataset_alias}, {algorithm_alias}: {execution_time:.4f} seconds")
                # print(model_name, dataset_alias, algorithm_alias)  # prueba
