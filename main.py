from src.utils import read_tsp_file, animate_iteration_paths
from src.aco import ant_colony_optimization
from src.ga import genetic_algorithm
import matplotlib.pyplot as plt
import csv
import os

def log_to_csv(filename, data, header):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)

def record_runs(coords, log_file, header, num_runs=10):
    for run in range(1, num_runs + 1): 
        print(f"\nRun {run}")

        #ACO
        aco_route, aco_len, aco_iteration_paths, aco_iter_lengths, _ = ant_colony_optimization(coords, use_2opt=False)
        log_to_csv(log_file, ["ACO", run, len(aco_iter_lengths), aco_len], header)

        #ACO + 2-opt
        aco2_route, aco2_len, aco2_iteration_paths, aco2_iter_lengths, _ = ant_colony_optimization(coords, use_2opt=True)
        log_to_csv(log_file, ["ACO-2opt", run, len(aco2_iter_lengths), aco2_len], header)

        #GA
        ga_route, ga_len, ga_iteration_paths, _, ga_current_lengths = genetic_algorithm(coords, use_2opt=False)
        log_to_csv(log_file, ["GA", run, len(ga_current_lengths), ga_len], header)

        #GA + 2-opt
        ga2_route, ga2_len, ga2_iteration_paths, _, ga2_current_lengths = genetic_algorithm(coords, use_2opt=True)
        log_to_csv(log_file, ["GA-2opt", run, len(ga2_current_lengths), ga2_len], header)

    print("\nAll runs complete. Results logged to", log_file)
    
    
if __name__ == "__main__":
    coords = read_tsp_file("data/berlin52.tsp")
    header = ["Algorithm", "Run", "Iterations", "Path Length"]
    log_file = "results_log.csv" 


    aco_route, aco_len, aco_iteration_paths, aco_iter_lengths, aco_best_so_far_lengths = ant_colony_optimization(coords, use_2opt=False)
    aco2_route, aco2_len, aco2_iteration_paths, aco2_iter_lengths, aco2_best_so_far_lengths = ant_colony_optimization(coords, use_2opt=True)

    ga_route, ga_len, ga_iteration_paths, ga_best_so_far_lengths, ga_current_lengths = genetic_algorithm(coords, use_2opt=False)
    ga2_route, ga2_len, ga2_iteration_paths, ga2_best_so_far_lengths, ga2_current_lengths = genetic_algorithm(coords, use_2opt=True)
    
    ani = animate_iteration_paths(
        coords,
        aco_paths=aco_iteration_paths,
        aco_best_dists=aco_best_so_far_lengths,
        aco_current_dists=aco_iter_lengths,
        ga_paths=ga_iteration_paths,
        ga_best_dists=ga_best_so_far_lengths,
        ga_current_dists=ga_current_lengths,
        aco2opt_paths=aco2_iteration_paths,
        aco2opt_best_dists=aco2_best_so_far_lengths,
        aco2opt_current_dists=aco2_iter_lengths,
        ga2opt_paths=ga2_iteration_paths,
        ga2opt_best_dists=ga2_best_so_far_lengths,
        ga2opt_current_dists=ga2_current_lengths
    )

    plt.show()
    
    
    #for logging of results.
    
    # record_runs(coords,log_file, header, num_runs=10)