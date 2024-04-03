import matplotlib.pyplot as plt


def visualize_fitness_measure(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()


def visualize_population(axs, subplot_x_idx, subplot_y_idx, X, Y, Z, population_x_coords, population_y_coords):
    contours = axs[subplot_x_idx, subplot_y_idx].contour(X, Y, Z, levels=10, cmap='viridis')
    axs[subplot_x_idx, subplot_y_idx].clabel(contours, inline=True, fontsize=8)
    axs[subplot_x_idx, subplot_y_idx].set_axis_off()
    axs[subplot_x_idx, subplot_y_idx].scatter(population_x_coords, population_y_coords, marker='.', color='red', s=10)


def visualize_direction_pso(axs, subplot_x_idx, subplot_y_idx, particles_pos, particles_v, global_best, global_best_index):
    axs[subplot_x_idx, subplot_y_idx].quiver(particles_pos[0], particles_pos[1], particles_v[0], particles_v[1], color='red', scale=10, width=0.003)
    axs[subplot_x_idx, subplot_y_idx].scatter(global_best[0], global_best[1], marker='*', color='green')
    axs[subplot_x_idx, subplot_y_idx].quiver(global_best[0], global_best[1], particles_v[0, global_best_index], particles_v[1, global_best_index],
               color='green', scale=12, width=0.003)
