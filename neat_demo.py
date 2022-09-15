################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from neat_controller import player_controller
import time as tm
import neat, visualize, pygame, pickle

experiment_name = 'NEAT_specialized_agent_7_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Initialize our parameters for specialized agent
enemy_level = 7
generations = 10

# initializes output directory and training simulation in individual evolution mode, for single static enemy.
OUTPUT_DIR = './' + experiment_name + '/'
TRAINING_ENV = Environment(experiment_name=experiment_name,
                enemies=[enemy_level],
                playermode="ai",
                player_controller=player_controller(),
                enemymode="static",
                level=1,
                speed="fastest")

def eval_genomes(genomes, config):
    global TRAINING_ENV
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness, player_life, enemy_life, time = TRAINING_ENV.play(pcont=net)
        genome.fitness = fitness

def quit_environment():
    pygame.display.quit() 
    pygame.quit()

def visualize_results(config, winner, stats, node_names=None):
    # Plot to show the structure of a network described by a genome.
    visualize.draw_net(config, winner, True, node_names=node_names, filename='nn_vis', file_dir=OUTPUT_DIR)
        # get_pruned_copy not present in DefaultGenome, therefore line below will not work
        # visualize.draw_net(config, winner, True, prune_unused=True)
    # Visualize to plot the best and average fitness vs. generation
    visualize.plot_stats(stats, ylog=False, view=True, file_dir=OUTPUT_DIR)
    # plot the change in species vs. generation
    visualize.plot_species(stats, view=True, file_dir=OUTPUT_DIR)

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to x amount of generations.
    winner = p.run(eval_genomes, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # Saving winner with pickle
    pickle.dump(winner_net, open(OUTPUT_DIR + 'winner_agent_nn', 'wb'))
    # Run final evaluation with winner
    eval_env = Environment(experiment_name=experiment_name,
                enemies=[enemy_level],
                playermode="ai",
                player_controller=player_controller(),
                enemymode="static",
                level=1,
                speed="normal")
    fitness, player_life, enemy_life, time = eval_env.play(pcont=winner_net)
    quit_environment()
    # Visualization of the experiment
    node_names = {0: 'Left', 1: 'Right', 2: 'Jump', 3: 'Shoot', 4: 'Release'}
    visualize_results(config, winner, stats, node_names=node_names)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)