import csv

from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.indicators.gd import GD
from pymoo.optimize import minimize

def optimize_and_save(ref_dirs, ref_point, dataset_name, algorithm_instance, algorithm_name, model_instance, model_name, problem, num_iterations):

    fitness_file = f"{dataset_name}_fitness_{algorithm_name}_maop_{model_name}.csv"
    solution_file = f'{dataset_name}_solution_{algorithm_name}_maop_{model_name}.csv'
    metrics_file = f'{dataset_name}_metrics_{algorithm_name}_maop_{model_name}.csv'

    gd = GD(ref_dirs)
    igd = IGD(ref_dirs)
    hv = HV(ref_point=ref_point)

    with open(fitness_file, 'w', newline='') as f, open(solution_file, 'w', newline='') as g, open(metrics_file, 'w', newline='') as h:
        writerFitness = csv.writer(f)
        writerPopulation = csv.writer(g)
        writerMetric = csv.writer(h)

        writerFitness.writerow(["ACC", "NFS", "MI", "MacroF1"])
        writerMetric.writerow(["GD", "IGD", "HV"])

        for i in range(num_iterations):
            res = minimize(problem, algorithm_instance, ('n_gen', 90), verbose=True, save_history=False)

            value_fitness = np.unique(res.F, axis=0)
            value_solution = np.unique(res.X, axis=0)

            metrics = np.array([gd(value_fitness), igd(value_fitness), hv(value_fitness)])
            writerMetric.writerow(anp.column_stack(metrics))

            writerFitness.writerows(value_fitness)
            writerPopulation.writerows(value_solution)

            writerMetric.writerow([''])
            writerFitness.writerow([''])
            writerPopulation.writerow([''])

            print(i, ' ')