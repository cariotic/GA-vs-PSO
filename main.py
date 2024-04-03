import matplotlib.pyplot as plt

from ga import *
from pso import *
from visualization import *

N_GENERATIONS = 20
POPULATION_SIZE = 100


def reshape_population_for_pso(population):
    tmp = [list(t) for t in population]
    return np.reshape(tmp, (2, -1))


if __name__ == '__main__':
    X, Y = np.meshgrid(np.arange(-5, 5, 0.05), np.arange(-5, 5, 0.05))
    Z = fitness_measure(X, Y)
    visualize_fitness_measure(X, Y, Z)

    # setting up initial population
    population_coords_ga = initialize_population_ga(POPULATION_SIZE)
    population_coords_pso = initialize_population_pso(POPULATION_SIZE)
    x_coords_ga = [item[0] for item in population_coords_ga]
    y_coords_ga = [item[1] for item in population_coords_ga]

    # setting up pso
    population_velocity = initialize_population_velocity(POPULATION_SIZE)
    individual_best_coords = population_coords_pso
    individual_best_values = fitness_measure(population_coords_pso[0], population_coords_pso[1])
    global_best_index = np.argmin(individual_best_values)
    global_best_value = min(individual_best_values)
    global_best_coords = individual_best_coords[:, global_best_index]

    fig, axs = plt.subplots(3, 2)
    fig.suptitle("Genetic Algorithm vs Particle Swarm Optimization", y=0.99)
    plt.subplots_adjust(top=0.85)
    plt.figtext(0.3, 0.9, "GA", fontsize=8)
    plt.figtext(0.7, 0.9, "PSO", fontsize=8)

    for i in range(N_GENERATIONS):
        # 1. genetic algorithm
        if i == 0:
            visualize_population(axs, 0, 0, X, Y, Z, x_coords_ga, y_coords_ga)
            axs[0, 0].set_title("initial state", fontsize=8)
        if i == N_GENERATIONS / 2:
            visualize_population(axs, 1, 0, X, Y, Z, x_coords_ga, y_coords_ga)
            axs[1, 0].set_title(f"after {int(N_GENERATIONS/2)} iterations", fontsize=8)
        if i == N_GENERATIONS - 1:
            visualize_population(axs, 2, 0, X, Y, Z, x_coords_ga, y_coords_ga)
            axs[2, 0].set_title(f"after {N_GENERATIONS} iterations", fontsize=8)

        new_population_ga = []
        fit = evaluate_fitness(population_coords_ga)
        for j in range(int(len(population_coords_ga) / 2)):
            # selecting parents
            parent1, parent2 = selection(population_coords_ga, fit)
            # crossover
            child1, child2 = crossover(parent1, parent2)
            new_population_ga.append(child1)
            new_population_ga.append(child2)
        population_coords_ga = new_population_ga
        # mutation
        mutated_idx = random.randint(0, int(len(population_coords_ga) - 1))
        mutated_subject = (population_coords_ga[mutated_idx][0] + random.random(), population_coords_ga[mutated_idx][1]
                           + random.random())
        population_coords_ga[mutated_idx][:] = mutated_subject

        x_coords_ga = [item[0] for item in population_coords_ga]
        y_coords_ga = [item[1] for item in population_coords_ga]


        # 2. particle swarm optimization
        if i == 0:
            visualize_population(axs, 0, 1, X, Y, Z, population_coords_pso[0], population_coords_pso[1])
            visualize_direction_pso(axs, 0, 1, population_coords_pso, population_velocity, global_best_coords, global_best_index)
            axs[0, 1].set_title("initial state", fontsize=8)
        if i == N_GENERATIONS / 2:
            visualize_population(axs, 1, 1, X, Y, Z, population_coords_pso[0], population_coords_pso[1])
            visualize_direction_pso(axs, 1, 1, population_coords_pso, population_velocity, global_best_coords, global_best_index)
            axs[1, 1].set_title(f"after {int(N_GENERATIONS/2)} iterations", fontsize=8)
        if i == N_GENERATIONS - 1:
            visualize_population(axs, 2, 1, X, Y, Z, population_coords_pso[0], population_coords_pso[1])
            visualize_direction_pso(axs, 2, 1, population_coords_pso, population_velocity, global_best_coords, global_best_index)
            axs[2, 1].set_title(f"after {N_GENERATIONS} iterations", fontsize=8)

        # updating velocities and positions
        population_velocity = update_velocity(population_coords_pso, population_velocity, individual_best_coords, global_best_coords)
        population_coords_pso = update_position(population_coords_pso, population_velocity)
        fitness_value = fitness_measure(population_coords_pso[0], population_coords_pso[1])
        # updating best value ever found by a particle
        individual_best_coords, individual_best_values = find_personal_best(POPULATION_SIZE, population_coords_pso,
                                                                            fitness_value, individual_best_values, individual_best_coords)
        # updating best value ever found globally
        global_best_value = min(individual_best_values)
        global_best_index, global_best_coords = find_global_best(individual_best_values, individual_best_coords)

    plt.show()




