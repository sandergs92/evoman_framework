from demo_controller import player_controller

# imports framework
import sys, os
import numpy as np
from numpy.random import randint
from numpy.random import rand
import random
sys.path.insert(0, 'evoman') 
from environment import Environment

experiment_name = 'simple_GA_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 0

# initializes training simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                enemies=[1],
                playermode="ai",
                player_controller=player_controller(n_hidden_neurons),
                enemymode="static",
                level=1,
                speed="fastest")

# set global training parameters
bit_length = env.get_num_sensors()
n_population = 20
n_generations = 10
value1 = -1
value2 = 1







