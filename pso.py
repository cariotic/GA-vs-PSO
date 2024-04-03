import random
from fitness_measure import *

C1 = 0.1
C2 = 0.1
ALPHA = 0.5


def initialize_population_pso(population_size):
    return np.random.rand(2, population_size) * 10 - 5

def initialize_population_velocity(population_size):
    return np.random.randn(2, population_size) * 0.1


def find_individual_best_values(x_coords, y_coords):
    return fitness_measure(x_coords[0], y_coords[1])


def find_global_best(individual_best_values, individual_best_coords):
    global_best_index = np.argmin(individual_best_values)
    return global_best_index, individual_best_coords[:, global_best_index]


def update_velocity(coords, population_velocity, personal_best, global_best):
    r1 = random.random()
    r2 = random.random()
    return ALPHA*population_velocity + C1*r1*(personal_best - coords) + C2*r2*(global_best.reshape(2, 1) - coords)


def update_position(coords, population_velocity):
    return coords + population_velocity


def find_personal_best(population_size, coords, fitness_value, personal_best_value, personal_best):
    for j in range(population_size):
        if fitness_value[j] < personal_best_value[j]:
            personal_best[:, j] = coords[:, j]
            personal_best_value[j] = fitness_value[j]
    return personal_best, personal_best_value


