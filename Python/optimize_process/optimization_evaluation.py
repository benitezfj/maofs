import csv
import numpy as np
import autograd.numpy as anp

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.optimize import minimize
from methods.objectives import featureSelectionMany

def optimize_and_save(ref_dirs, dataset_name, algorithm_instance, algorithm_name, model_instance, model_name, 
                      problem, train_features, train_classes, val_features, val_classes, filter_info, num_iterations):
    print("Evaluating...", " Model: ", model_name, " Algorithm: ", algorithm_name, " Dataset: ", dataset_name)

    fitness_file = f'{dataset_name}_fitness_{algorithm_name}_{model_name}.csv'
    solution_file = f'{dataset_name}_solution_{algorithm_name}_{model_name}.csv'
    metrics_file = f'{dataset_name}_metrics_{algorithm_name}_{model_name}.csv'
    evaluation_file = f'{dataset_name}_evaluation_{algorithm_name}_{model_name}.csv'
    
    ref_point = ref_dirs.max(axis=0)
    
    gd = GD(ref_dirs)
    igd = IGD(ref_dirs)
    hv = HV(ref_point=ref_point)

    with open(fitness_file, 'w', newline='') as f, open(solution_file, 'w', newline='') as g, open(metrics_file, 'w', newline='') as h, open(evaluation_file,'w', newline='') as i:
        writerFitness = csv.writer(f)
        writerPopulation = csv.writer(g)
        writerMetric = csv.writer(h)
        writerEval = csv.writer(i)

        writerFitness.writerow(["Recall","NFS","SU","MacroF1"])
        writerMetric.writerow(["GD","IGD","HV"])
        writerEval.writerow(["Recall","NFS","SU","MacroF1","ACC"])

        for i in range(num_iterations):
            res = minimize(problem, algorithm_instance, ('n_gen', 100), seed=i, verbose=True, save_history=False)

            # value_fitness = np.unique(res.F, axis=0)
            writerFitness.writerows(res.F)

            # value_solution = np.unique(res.X, axis=0)
            writerPopulation.writerows(res.X)
            for population in res.X:
                evaluation=featureSelectionMany(x=population, 
                                                X_train=train_features,
                                                X_test=val_features,
                                                y_train=train_classes,
                                                y_test=val_classes,
                                                mutual_info=filter_info,
                                                estimator=model_instance)
                writerEval.writerow(evaluation)

            metrics = np.array([gd(res.F), igd(res.F), hv(res.F)])
            writerMetric.writerow(anp.column_stack(metrics))

            writerMetric.writerow([''])
            writerFitness.writerow([''])
            writerPopulation.writerow([''])
            writerEval.writerow([''])

            print(i, ' ')