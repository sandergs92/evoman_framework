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
import neat

experiment_name = 'neat_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes training simulation in individual evolution mode, for single static enemy.
TRAINING_ENV = Environment(experiment_name=experiment_name,
                enemies=[1],
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

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    eval_env = Environment(experiment_name=experiment_name,
                enemies=[1],
                playermode="ai",
                player_controller=player_controller(),
                enemymode="static",
                level=1,
                speed="normal")
    fitness, player_life, enemy_life, time = eval_env.play(pcont=winner_net)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)