################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os, random, copy, math, pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Init experiment
enemy = 4
n_experiments = 10
experiment_name = 'simple_ea_specialized_agent_' + str(enemy)
OUTPUT_DIR = './' + experiment_name + '/'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# global training parameters
n_population = 100
n_generations = 30
min_value = -1
max_value = 1
# Neural net parameters
n_hidden_neurons = 3
    # initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[enemy],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons+(n_hidden_neurons+1)*5
# Operator parameters
k_individuals = round(n_population / 4)
mutation_rate = 10 / n_weights

def initialise_population()->np.ndarray:
    """Returns a population with shape """
    return np.random.uniform(min_value, max_value, (n_population, n_weights))

def evalualate_pop(population: np.ndarray):
    """Function to evaluate current population """
    fit_pop = np.array([])
    for agent in population:
        fitness, player_life, enemy_life, time = env.play(pcont=agent)
        fit_pop = np.append(fit_pop, fitness)
    return fit_pop
    
def tournament_selection(population: np.ndarray, fitness_population, k:int)->np.ndarray:
    """Function takes a population and a k individuals as input, hosts a tournament and returns population with winners."""
    new_pop = np.ndarray(shape=population.shape)
    for i in range(len(population)):
        random_indices = random.sample(range(len(population)), round(k))
        best_individual = population[random_indices][np.argmax(fitness_population[random_indices])]
        new_pop[i] = best_individual
    return new_pop

def two_point_crossover(population: np.ndarray)->np.ndarray:
    """Function that uses two-point crossover for population recombination, returns mutated population."""
    new_pop = np.ndarray(shape=population.shape)
    for i in range(n_population // 2):
        child_1 = copy.copy(population[i])
        child_2 = copy.copy(population[i+1])

        sorted_rand_indices = sorted(random.sample(range(round(n_weights/3), n_weights - round(n_weights/3)), 2))
        gene_seq_1 = child_1[sorted_rand_indices[0]:sorted_rand_indices[1]]
        gene_seq_2 = child_2[sorted_rand_indices[0]:sorted_rand_indices[1]]
        child_1[sorted_rand_indices[0]:sorted_rand_indices[1]] = gene_seq_2
        child_2[sorted_rand_indices[0]:sorted_rand_indices[1]] = gene_seq_1

        new_pop[i] = child_1
        new_pop[i+1] = child_2
    return new_pop

def mutation(population: np.ndarray, m_rate:int)->np.ndarray:
    """Mutates population according to the given mutation rate, returns mutated population."""
    new_pop = np.ndarray(shape=population.shape)
    for i in range(len(population)):
        random_genes = np.array(random.sample(range(n_weights), math.ceil(m_rate * n_weights)))
        mutated_agent = population[i]
        np.put(mutated_agent, random_genes, np.random.uniform(low=-1.0, high=1.0, size=random_genes.shape))
        new_pop[i] = mutated_agent
    return new_pop

def evo_run():
    """Evolutionary Algorithm loop function"""
    experiment_stats = dict()
    for e in range(0, n_experiments):
        stats = dict()
        current_pop = initialise_population()
        for g in range(0, n_generations):
            print('Experiment run:', e, 'Generation:', g)
            # Evaluate
            current_pop_f = evalualate_pop(current_pop)
            # Add entry to stats
            stats[g] = {
                'max_f' : np.max(current_pop_f),
                'min_f' : np.min(current_pop_f),
                'avg_f' : np.mean(current_pop_f),
                'std_f' : np.std(current_pop_f),
                'pop' : current_pop,
                'pop_f' : current_pop_f,
                'best_individual' : current_pop[np.argmax(current_pop_f)]
            }
            print('Max f:', np.max(current_pop_f), ' Min f:', np.min(current_pop_f), ' Avg f:', np.mean(current_pop_f), ' std_f f:', np.std(current_pop_f))
            # Parent selection
            current_pop = tournament_selection(current_pop, current_pop_f, k_individuals)
            # Cross-over
            current_pop = two_point_crossover(current_pop)
            # Mutation
            current_pop = mutation(current_pop, mutation_rate)
        experiment_stats[e] = stats
    pickle.dump(experiment_stats, open(OUTPUT_DIR + 'simple_ea_experiment_runs', 'wb'))
    # visualize.plot_stats(stats, simple_ga=True, file_dir=OUTPUT_DIR)

evo_run()