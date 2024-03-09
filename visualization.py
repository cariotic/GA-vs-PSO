import matplotlib.pyplot as plt


def visualize_fitness_measure(X, Y, Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()


def visualize_population(X, Y, Z, population_x_coords, population_y_coords):
    fig2, ax2 = plt.subplots()
    contours = ax2.contour(X, Y, Z, levels=10, cmap='viridis')
    ax2.clabel(contours, inline=True, fontsize=8)
    ax2.set_axis_off()
    plt.scatter(population_x_coords, population_y_coords, marker='.', color='red')
    #plt.show()


def visualize_direction_pso(particles_pos, particles_v, global_best, global_best_index):
    plt.quiver(particles_pos[0], particles_pos[1], particles_v[0], particles_v[1], color='red', scale=10, width=0.003)
    plt.scatter(global_best[0], global_best[1], marker='*', color='green')
    plt.quiver(global_best[0], global_best[1], particles_v[0, global_best_index], particles_v[1, global_best_index],
               color='green', scale=10, width=0.003)
