import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_tsp_file(filename):
    coords = []
    with open(filename, 'r') as file:
        start = False
        for line in file:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                start = True
                continue
            if start:
                if line == "EOF" or line == "":
                    break
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return coords

#meant for 2D Euclidean only!
def build_matrix(coords):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                xi, yi = coords[i]
                xj, yj = coords[j]
                dij = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                matrix[i][j] = int(dij + 0.5)
    return matrix

def calculate_length(route, D):
    return sum(D[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))


#for visualising the runs.
def animate_iteration_paths(coords,
                            aco_paths, aco_best_dists, aco_current_dists,
                            ga_paths, ga_best_dists, ga_current_dists,
                            aco2opt_paths, aco2opt_best_dists, aco2opt_current_dists,
                            ga2opt_paths, ga2opt_best_dists, ga2opt_current_dists):
    #color and styling
    plot_color = '#ffffff'
    line_color = '#08C2FF'
    canvas_color = '#0b0c10'
    text_color = '#ffffff'

    coords = np.array(coords)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 5))
    fig.patch.set_facecolor(canvas_color)
    fig.canvas.manager.set_window_title("TSP Algorithm Comparison")
    fig.canvas.manager.toolbar.pack_forget() 

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(canvas_color)

    def draw_panel(ax, title, tour_edges, dist_current, dist_best, label, frame_index):
        ax.clear()
        ax.set_facecolor(canvas_color)
        ax.set_title(title, color=text_color, fontsize=14)
        ax.plot(coords[:, 0], coords[:, 1], 'o', color=plot_color)

        for start, end in tour_edges:
            if start != end:
                ax.plot([coords[start][0], coords[end][0]],
                        [coords[start][1], coords[end][1]], color=line_color)

        ax.axis('equal')
        ax.axis('off')
        info = f"{label}: {frame_index+1}\nCurrent: {dist_current:.2f}\nBest: {dist_best:.2f}"
        ax.text(0.5, -0.1, info, transform=ax.transAxes, ha='center', color=text_color, fontsize=11)

    def update(i):
        aco_frame = min(i, len(aco_paths) - 1)
        ga_frame = min(i, len(ga_paths) - 1)
        aco2_frame = min(i, len(aco2opt_paths) - 1)
        ga2_frame = min(i, len(ga2opt_paths) - 1)

        draw_panel(ax1, "Ant Colony Optimization",
                   aco_paths[aco_frame],
                   aco_current_dists[aco_frame],
                   aco_best_dists[aco_frame],
                   "Iter", aco_frame)

        draw_panel(ax2, "Genetic Algorithm",
                   ga_paths[ga_frame],
                   ga_current_dists[ga_frame],
                   ga_best_dists[ga_frame],
                   "Gen", ga_frame)

        draw_panel(ax3, "ACO + 2-opt",
                   aco2opt_paths[aco2_frame],
                   aco2opt_current_dists[aco2_frame],
                   aco2opt_best_dists[aco2_frame],
                   "Iter", aco2_frame)

        draw_panel(ax4, "GA + 2-opt",
                   ga2opt_paths[ga2_frame],
                   ga2opt_current_dists[ga2_frame],
                   ga2opt_best_dists[ga2_frame],
                   "Gen", ga2_frame)

    max_frames = max(
        len(aco_paths),
        len(ga_paths),
        len(aco2opt_paths),
        len(ga2opt_paths)
    )
    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=300, repeat=False)
    return ani

