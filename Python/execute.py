from networktrafficoriginators import optimize_model

model_names = ['gnb', 'knn']
dataset_aliases = ["linux_memory", "linux_disk", "linux_process", "network", "win7", "win10"]
algorithm_aliases = ["NSGA-II", "NSGA-III", "RVEA", "MOEA/D"]
num_iterations = 12


if __name__ == "__main__":
  for model_name in model_names:
    for dataset_alias in dataset_aliases:
      for algorithm_alias in algorithm_aliases:
        algorithm_name = algorithm_alias
        dataset_name = dataset_alias
        optimize_model(model_name, dataset_name, algorithm_name, num_iterations)