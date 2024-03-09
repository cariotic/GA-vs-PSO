import random
from fitness_measure import *
from visualization import *


def fitness_measure2(fitness_scores):
    p = 0.1
    wmax = max(fitness_scores)
    wmin = min(fitness_scores)
    fit2 = []
    for ind_fit in fitness_scores:
        ind_fit2 = ((1 - p) * ind_fit + p * wmin - wmax) / (wmin - wmax)
        fit2.append(ind_fit2)
    return fit2


def evaluate_fitness(population):
    fitness_scores1 = []
    for x, y in population:
        fitness_scores1.append(fitness_measure(x, y))
    fitness_scores2 = fitness_measure2(fitness_scores1)
    return fitness_scores2


def create_prob(fitness_scores):
    return[individual_fit/sum(fitness_scores) for individual_fit in fitness_scores]


def selection(population, fitness_scores):
    return random.choices(population, weights=create_prob(fitness_scores), k=2)


def crossover(parent1, parent2):
    child1 = [0]*2
    child2 = [0]*2
    dw1 = parent2[0] - parent1[0]
    dw2 = parent2[1] - parent1[1]
    alpha = 0.2 * (np.random.rand()-0.5)
    child1[:] = [parent1[0] + dw1*alpha, parent1[1] + dw2*alpha]
    child2[:] = [parent2[0] - dw1*alpha, parent2[1] - dw2*alpha]
    return child1, child2


# if __name__ == '__main__':
#     # X, Y = np.meshgrid(np.arange(-5, 5, 0.05), np.arange(-5, 5, 0.05))
#     # Z = fitness_measure(X, Y)
#     # visualize_fitness_measure(X, Y, Z)
#     #
#     # # setting up initial population
#     # population = initialize_population(POPULATION_SIZE)
#     # x_coords = [item[0] for item in population]
#     # y_coords = [item[1] for item in population]
#     # visualize_population(X, Y, Z, x_coords, y_coords)
#
#     for i in range(N_GENERATIONS):
#         new_population = []
#         fit = evaluate_fitness(population)
#         for j in range(int(len(population)/2)):
#             # selecting parents
#             parent1, parent2 = selection(population, fit)
#             # crossover
#             child1, child2 = crossover(parent1, parent2)
#             new_population.append(child1)
#             new_population.append(child2)
#         population = new_population
#         visualize_population(X, Y, Z, x_coords, y_coords)
#         # mutation
#         mutated_idx = random.randint(0, int(len(population) - 1))
#         mutated_subject = (population[mutated_idx][0] + random.random(), population[mutated_idx][1] + random.random())
#         population[mutated_idx][:] = mutated_subject
#
#         x_coords = [item[0] for item in population]
#         y_coords = [item[1] for item in population]
#         visualize_population(X, Y, Z, x_coords, y_coords)
