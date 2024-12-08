from networktrafficoriginators import optimize_model

model_names = ["gnb"]
dataset_aliases = ["breast_cancer_chin_R1_K4_s281635"]
algorithm_aliases = ["RVEA"]
# model_names = ["gnb", "rf", "knn"]
# dataset_aliases = ["linux_memory", "linux_disk", "linux_process", "linux_network", "win7", "win10"]
# algorithm_aliases = ["NSGA-II", "NSGA-III", "RVEA", "MOEAD"]
# num_iterations = 20

if __name__ == "__main__":
    for model_name in model_names:
        for dataset_alias in dataset_aliases:
            for algorithm_alias in algorithm_aliases:
                optimize_model(model_name, dataset_alias, algorithm_alias)
                # print(model_name, dataset_alias, algorithm_alias)  # prueba
