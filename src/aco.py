import numpy as np
import random
from src.utils import build_matrix, calculate_length
from src.two_opt import two_opt


def route_to_edges(route):
    """Convert a route (list of city indices) into list of edges (pairs of cities)."""
    return [(route[i], route[(i + 1) % len(route)]) for i in range(len(route))]


def ant_colony_optimization(coords, population_size=100, max_iterations=1000, evaporation_rate=0.5,
                             alpha=1, beta=2, q=50, artificial_pheromone=1.0,
                             no_improvement_limit=80, use_2opt=False):
    n = len(coords)
    D = build_matrix(coords)

    #Initialize pheromone and heuristic
    Tau = np.full((n, n), artificial_pheromone)
    Eta = 1 / (D + np.eye(n))  #Add diagonal to prevent div by zero
    np.fill_diagonal(Eta, 0)

    best_route = None
    best_length = float('inf')
    no_improvement_counter = 0
    last_improvement_iteration = 0

    route_edges_history = []    #Edge lists for animation
    iter_best_lengths = []      #Current best in iteration
    best_so_far_lengths = []    #Cumulative best-so-far
    avg_lengths = []            #Avg length per iteration

    for iteration in range(max_iterations):
        all_routes = []
        all_lengths = []

        #Each ant constructs a route
        for ant in range(population_size):
            unvisited = list(range(n))
            current = random.choice(unvisited)
            route = [current]
            unvisited.remove(current)

            while unvisited:
                probabilities = []
                for city in unvisited:
                    tau = Tau[current][city] ** alpha
                    eta = Eta[current][city] ** beta
                    probabilities.append(tau * eta)

                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_city = random.choices(unvisited, weights=probabilities, k=1)[0]

                route.append(next_city)
                unvisited.remove(next_city)
                current = next_city

            all_routes.append(route)
            length = calculate_length(route, D)
            all_lengths.append(length)

        #Update pheromones
        Tau *= (1 - evaporation_rate)
        for route, length in zip(all_routes, all_lengths):
            delta = q / length
            for i in range(n):
                a, b = route[i], route[(i + 1) % n]
                Tau[a][b] += delta
                Tau[b][a] += delta

        #Track stats
        avg_lengths.append(np.mean(all_lengths))
        current_best_index = np.argmin(all_lengths)
        best_candidate = all_routes[current_best_index]
        if use_2opt:
            current_best_route, current_best_length = two_opt(best_candidate, D)
        else:
            current_best_route = best_candidate
            current_best_length = all_lengths[current_best_index]

        iter_best_lengths.append(current_best_length)
        route_edges_history.append(route_to_edges(current_best_route))

        if current_best_length < best_length - 1e-6:
            best_length = current_best_length
            best_route = current_best_route.copy()
            no_improvement_counter = 0
            last_improvement_iteration = iteration
        else:
            no_improvement_counter += 1

        print(f"ACO Iter {iteration+1} - Current Best: {current_best_length:.2f} | Global Best: {best_length:.2f} | No Improvement: {no_improvement_counter}")

        if no_improvement_counter >= no_improvement_limit:
            break

    #Compute best-so-far list
    best_so_far = float('inf')
    for l in iter_best_lengths:
        best_so_far = min(best_so_far, l)
        best_so_far_lengths.append(best_so_far)

    final_idx = last_improvement_iteration + 1
    return (
        best_route,
        best_length,
        route_edges_history[:final_idx],
        iter_best_lengths[:final_idx],
        best_so_far_lengths[:final_idx]
    )
