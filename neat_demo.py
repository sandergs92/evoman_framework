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
import neat, pygame, pickle

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize our parameters for specialized agent
enemy = 1
generations = 30

# Initialize experiment
n_experiments = 10
experiment_name = 'NEAT_specialized_agent_' + str(enemy)
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes output directory and training simulation in individual evolution mode, for single static enemy.
OUTPUT_DIR = './' + experiment_name + '/'
TRAINING_ENV = Environment(experiment_name=experiment_name,
                enemies=[enemy],
                playermode="ai",
                player_controller=player_controller(),
                enemymode="static",
                level=2,
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

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    experiment_stats = dict()
    for e in range(n_experiments):
        statistics = dict()
        print("EXPERIMENT RUN:", e)
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

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
        # Saving winner and stats in dict
        statistics = {
            'population': p,
            'winner_net': winner_net,
            'winning_genome': winner,
            'stats': stats
        }
        experiment_stats[e] = statistics
    quit_environment()
    pickle.dump(experiment_stats, open(OUTPUT_DIR + 'neat_experiment_runs', 'wb'))

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)