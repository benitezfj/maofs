from networktrafficoriginators import optimize_model

model_names = ['gnb', 'rf', 'knn']
dataset_aliases = ["linux_memory", "linux_disk", "linux_process", "network", "win7", "win10"]
algorithm_aliases = ["NSGA-II", "NSGA-III", "RVEA", "MOEAD"]
num_iterations = 20


if __name__ == "__main__":
  for model_name in model_names:
    for dataset_alias in dataset_aliases:
      for algorithm_alias in algorithm_aliases:
        optimize_model(model_name, dataset_alias, algorithm_alias, num_iterations)
        # print(model_name, dataset_alias, algorithm_alias, num_iterations) prueba