import random
from src.utils import build_matrix, calculate_length
from src.two_opt import two_opt

def is_valid_route(route, n):
    return len(route) == n and len(set(route)) == n

def route_to_edges(route):
    return [(route[i], route[(i + 1) % len(route)]) for i in range(len(route))]

def ordered_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child1 = [None] * size
    child1[a:b] = p1[a:b]
    fill = [gene for gene in p2 if gene not in child1]
    idx = 0
    for i in range(size):
        if child1[i] is None:
            child1[i] = fill[idx]
            idx += 1

    child2 = [None] * size
    child2[a:b] = p2[a:b]
    fill = [gene for gene in p1 if gene not in child2]
    idx = 0
    for i in range(size):
        if child2[i] is None:
            child2[i] = fill[idx]
            idx += 1

    return child1, child2

def swap_mutation(individual):
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]

def genetic_algorithm(coords, population_size=100, generations=1000,
                      crossover_rate=0.8, mutation_rate=0.1, elite_ratio=0.1,
                      no_improvement_limit=80, use_2opt=False):

    n = len(coords)
    D = build_matrix(coords)

    elite_size = int(population_size * elite_ratio)
    population = [random.sample(range(n), n) for _ in range(population_size)]

    best_route = None
    best_length = float('inf')
    no_improvement_counter = 0
    last_improvement_iteration = 0

    route_edges_history = []
    best_lengths = []
    current_lengths = []

    for gen in range(generations):
        fitness = [1 / calculate_length(ind, D) for ind in population]
        probs = [f / sum(fitness) for f in fitness]

        selected = [population[random.choices(range(population_size), weights=probs)[0]]
                    for _ in range(population_size)]

        ranked = sorted(population, key=lambda x: calculate_length(x, D))
        elite = ranked[:elite_size]

        next_gen = elite.copy()
        while len(next_gen) < population_size:
            p1, p2 = random.sample(selected, 2)
            if random.random() < crossover_rate:
                c1, c2 = ordered_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if is_valid_route(c1, n): next_gen.append(c1)
            else: next_gen.append(random.sample(range(n), n))

            if len(next_gen) < population_size:
                if is_valid_route(c2, n): next_gen.append(c2)
                else: next_gen.append(random.sample(range(n), n))

        for i in range(elite_size, population_size):
            if random.random() < mutation_rate:
                swap_mutation(next_gen[i])

        population = next_gen

        current_best = min(population, key=lambda ind: calculate_length(ind, D))
        if use_2opt:
            current_best, current_len = two_opt(current_best, D)
        else:
            current_len = calculate_length(current_best, D)

        if current_len < best_length - 1e-6:
            best_length = current_len
            best_route = current_best.copy()
            no_improvement_counter = 0
            last_improvement_iteration = gen
        else:
            no_improvement_counter += 1

        route_edges_history.append(route_to_edges(current_best))
        current_lengths.append(current_len)
        best_lengths.append(best_length)

        print(f"GA Gen {gen+1} | Current: {current_len:.2f} | Best: {best_length:.2f} | No Improvement: {no_improvement_counter}")

        if no_improvement_counter >= no_improvement_limit:
            break

    final_idx = last_improvement_iteration + 1
    return (
        best_route,
        best_length,
        route_edges_history[:final_idx],
        best_lengths[:final_idx],
        current_lengths[:final_idx]
    )
