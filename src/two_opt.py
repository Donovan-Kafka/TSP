from src.utils import build_matrix, calculate_length

#modified 2opt to use with ga and aco--------------------------------------------------------------------------------------------
def two_opt(route, dist_matrix):
    best = route.copy()
    best_dist = calculate_length(best, dist_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:  # Skip adjacent
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                new_dist = calculate_length(new_route, dist_matrix)
                if new_dist < best_dist:
                    best = new_route
                    best_dist = new_dist
                    improved = True
        route = best
    return best, best_dist
#---------------------------------------------------------------------------------------------------------------------------------

# 2-opt algorithm
def algo2opt(coords, dist_matrix):
    # Initialise the first path 
    n = len(coords)
    currentBestPath = []
    for x in range(n):
        currentBestPath.append(x)
    currentBestPathDist = calculate_length(currentBestPath, dist_matrix)

    # Start 2-opt algorithm
    rounds = 0
    while True:
        # Round Starting
        hasImprovment = False

        # Round Body
        for v1 in range(1, n-1, 1):
            for v2 in range(v1+1, n, 1):
                part1 = []
                for i in range(v1):
                    part1.append(currentBestPath[i])
                
                part2 = []
                for i in range(v2, v1-1, -1):
                    part2.append(currentBestPath[i])
                
                part3 = []
                for i in range(v2+1, n, 1):
                    part3.append(currentBestPath[i])

                newPath = part1 + part2 + part3
                newPathDist = calculate_length(newPath, dist_matrix)

                if currentBestPathDist > newPathDist:
                    currentBestPathDist = newPathDist
                    currentBestPath = newPath
                    hasImprovment = True
                    # print the new path (if need)
                
                if hasImprovment:
                    break

            if hasImprovment:
                break

        # Round Ending
        rounds += 1
        if not hasImprovment:
            break
    
    return (currentBestPath, currentBestPathDist)
