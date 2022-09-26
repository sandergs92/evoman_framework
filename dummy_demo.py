################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports
import sys, os
import numpy.random as npr
import random
import operator
from numpy.random import randint
import numpy as np 
from environment import Environment
from demo_controller import player_controller

sys.path.insert(0, 'evoman')

experiment_name = 'simple_GA_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# global training parameters
n_hidden_neurons = 3
n_population = 10
n_generations = 30
min_value = -1
max_value = 1
tournament = True

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=1,
                  speed="fastest")

def evalualate_pop(population: np.ndarray) -> list:
    '''
    Tests given population on enemy, returns fitness values.
    '''
    res_pop = []
    for pop in population:
        fitness, player_life, enemy_life, time = env.play(pcont=pop)
        res_pop.append(fitness)
    return res_pop

def selection_prob(fit_val: list, sigma_scaling:bool=False) -> list:
    total_fit = sum(fit_val)
    std_fit = np.std(fit_val)
    avg_fit = np.mean(fit_val)
    # apply sigma scaling
    if sigma_scaling:
        sigma_fit = [np.max(fitness-(avg_fit-(2*std_fit)),0) for fitness in range(len(fit_val))]
        # calculate sum of sigma values to calculate selection probabilities
        total_sigma = np.sum(sigma_fit)
        select_prob = [sigma_fit[i]/total_sigma for i in range(fit_val)]
    # apply normal scaling
    else:
        select_prob = [(fit_val[i]/total_fit) for i in range(fit_val)]
    return select_prob

def tournament_selection(population: list, pop_dict: dict, pool_size: int=10, n_candidates: int=2) -> list:
    '''
    Returns the indices of the n candidates with the best fitness from the given pool size.
    '''
    pool = random.choices(population, k=pool_size)
    pool_fitness = [pop_dict[agent] for agent in pool]
    best_fitness = 0
    c_candidates = 1
    best_candidates = []
    while c_candidates <= n_candidates:
        for agent, fitness in zip(pool, pool_fitness):
            if fitness > best_fitness:
                best_fitness = fitness
                best_agent = agent
        index = population.index(best_agent)
        best_candidates.append(index)
        pool.remove(best_agent)
        pool_fitness.remove(best_fitness)
    return best_candidates

def roulette_selection(fit_val: list, n_parents: int=2, sigma_scaling: bool=False) -> list:
    select_prob = selection_prob(fit_val, sigma_scaling)
    c_parents = 1
    pool = []
    while c_parents <= n_parents:
        candidate = fit_val[npr.choice(len(fit_val), p=select_prob)]
        index = fit_val.index(candidate)
        pool.append(index)
        c_parents += 1
    return pool

def single_point_crossover(parent_1: list, parent_2: list) -> tuple(list):
    split = randint(1, len(parent_1)-1)
    child_1 = np.append(parent_1[:split], parent_2[split:])
    child_2 = np.append(parent_2[:split], parent_1[split:])
    return child_1, child_2

def two_point_crossover(parent_1: list, parent_2: list, n_splits: int=3) -> tuple(list):
    split_points = random.sample(range(1, len(parent_1)-1), n_splits)
    split_points.sort()
    for split in split_points:
        child_1, child_2 = single_point_crossover(parent_1, parent_2, split)
    return child_1, child_2

def uniform_crossover(parent_1: list, parent_2: list, p_uni: float=0.5) -> tuple(list):
    for gene in range(len(parent_1)):
        if p_uni > np.random.uniform():
            child_1 = parent_1[gene]
            child_2 = parent_2[gene]
        else:
            child_2 = parent_1[gene]
            child_1 = parent_2[gene]
    return child_1, child_2

def multi_parent_crossover():
    pass

def cross_over(p1: list, p2: list, n_splits: int=3, p_uni: float=0.5,
               single: bool=False, two_point: bool=False, uniform: bool=False, multi: bool=False) -> tuple(list):
    '''
    Uses the given parent selection method and then applies the given crossover 
    method to create offspring
    '''
    if single:
        return single_point_crossover(p1, p2)
    elif two_point:
        return two_point_crossover(p1, p2, n_splits)
    else:
        return uniform_crossover(p1, p2, p_uni)

def elite_pop(candidates: dict, n_agents: int=1) -> list:
    '''
    returns agents with best fitness in dictionary
    '''
    elite_pool = []
    for i in range(n_agents):
        elite = max(candidates.iteritems(), key=operator.itemgetter(1))[0]
        elite_pool.append(elite)
        candidates.pop(elite, 'Key not present in dictionary.')
    return elite_pool

def mutation(gene: list) -> list:
    mutate_rate = 1/len(gene)
    for chrom in gene:
        if random.uniform(0, 1) < mutate_rate:
            chrom =  np.random.uniform(min_value, max_value)
    return gene

## Start evolution algorithm
def evo_run(n_generations: int, n_population: int, n_hidden_neurons: int):
    n_weights = (env.get_num_sensors()+1)*n_hidden_neurons+(n_hidden_neurons+1)*5 # weights and biases needed for NN
    init_pop = np.random.uniform(min_value, max_value, (n_population, n_weights)) # initialize a random population
    for run in range(1, n_generations):
        mydict = dict()
        fit_val = evalualate_pop(init_pop)

        # store agents and their fitness in dictionary
        for agent, fitness in zip(init_pop, fit_val):
            mydict[agent] = fitness
        
        # collect statistics
        best_index = np.argmax(fit_val)
        best_fit = init_pop[best_index]
        std_fit = np.std(fit_val)
        avg_fit = np.mean(fit_val)

        # store current generation solutions
        solutions = [init_pop, fit_val]
        env.update_solutions(solutions)
        print(f'Generation: {run}, best fitness: {best_fit}, standard deviation: {std_fit}, average fitness: {avg_fit}.')

        # breed new generation
        elite = elite_pop(mydict)
        new_pop = [elite]
        lower_bound = round((n_population / 4) - len(elite))
        random_new = np.random.uniform(min_value, max_value, (lower_bound, n_weights))
        new_pop.extend(random_new)
        for i in range(n_population - len(new_pop)):
            if tournament:
                parent_1 = tournament_selection(init_pop, mydict)
                parent_2 = tournament_selection(init_pop, mydict)
            else:
                parent_1 = roulette_selection(fit_val)
                parent_2 = roulette_selection(fit_val)
        
        # apply crossover
        child_1, child_2 = cross_over(parent_1, parent_2, two_point=True)
        new_pop.extend([child_1, child_2])

        # apply mutation
        for gene in new_pop:
            gene = mutation(gene)
    
        # restart loop with new population
        new_pop = init_pop
    return new_pop

test = evo_run(n_generations, n_population, n_hidden_neurons)
