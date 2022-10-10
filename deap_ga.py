from math import factorial
import sys, os, random, copy, math, pickle, visualize

import matplotlib.pyplot as plt
import numpy as np

from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Init experiment
experiment_name = 'deap_generalized_agent'
OUTPUT_DIR = './' + experiment_name + '/'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# global training parameters
n_experiments = 10
n_population = 100
n_generations = 300
min_value = -1
max_value = 1
# Neural net parameters
n_hidden_neurons = 10
    # initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes",
                  randomini="yes",
                  clockprec="medium",
                  enemies=[7, 8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons+(n_hidden_neurons+1)*5
# DEAP parameters
n_objectives = 3
n_partitions = 12

def eval_individual(individual):
    """Function to evaluate current individual """
    fitness, player_life, enemy_life, time = env.play(pcont=individual)
    x1 = fitness
    x2 = (player_life - enemy_life) - np.log(time)
    x3 = player_life - np.log(time)
    print((x1, x2, x3))
    return (x1, x2, x3)

# Create uniform reference point
ref_points = tools.uniform_reference_points(n_objectives, n_partitions)

# Create classes
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

# Toolbox initialization
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.uniform, min_value, max_value)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_weights)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=min_value, up=max_value, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=min_value, up=max_value, eta=20.0, indpb=1.0/n_weights)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
##

def main(seed=None):
    for e in range(0, n_experiments):
        print("EXPERIMENT RUN: ", e)
        experiment_stats = dict()
        hof = 0
        hof_f = (0.0, 0.0, 0.0)
        random.seed(seed)

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = toolbox.population(n=n_population)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            if ind.fitness.values[0] > hof_f[0] and ind.fitness.values[1] > hof_f[1] and ind.fitness.values[2] > hof_f[2]:
                hof = ind
                hof_f = ind.fitness.values

        # Compile statistics about the population
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)
        experiment_stats[0] = pop

        # Begin the generational process
        for idx, gen in enumerate(range(1, n_generations)):
            offspring = algorithms.varAnd(pop, toolbox, 1.0, 1.0)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                if ind.fitness.values[0] > hof_f[0] and ind.fitness.values[1] > hof_f[1] and ind.fitness.values[2] > hof_f[2]:
                    hof = ind
                    hof_f = ind.fitness.values

            # Select the next generation population from parents and offspring
            pop = toolbox.select(pop + offspring, n_population)

            # Compile statistics about the new population
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
            experiment_stats[idx + 1] = pop

        experiment_stats['logbook'] = logbook
        experiment_stats['hof'] = hof
        experiment_stats['hof_f'] = hof_f
        pickle.dump(experiment_stats, open(OUTPUT_DIR + 'deap_stats_object_' + str(e), 'wb'))

if __name__ == "__main__":
    main()