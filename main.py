import numpy as np
import matplotlib.pyplot as plt
import argparse
from play import play


def get_labyrinth(labyrinth_file):
    labyrinth = []
    with open(labyrinth_file, 'r') as f:
        for line in f:
            labyrinth.append(line)
    labyrinth = np.array([list(line)[:-1] for line in labyrinth])
    return labyrinth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help='FIle path to labyrinth', default='data/labyrinth1.txt', type=str)
    parser.add_argument("-e", "--epsilon", help="Epsilon value", default=0.1, type=float)
    parser.add_argument("-a", "--alpha", help='Alpha value fro TD-SARSA algorithm', default='0.5', type=float)
    parser.add_argument("-g", "--gamma", help="Gamma value for TD-SARSA algorithm", default=0.9, type=float)
    parser.add_argument("-n", "--num_trials", help="Number of trials/episodes", default=50000, type=int)
    args = parser.parse_args()
    labyrinth_file = args.file_path
    eps = args.epsilon
    alpha = args.alpha
    gamma = args.gamma
    n_trials = args.num_trials

    labyrinth = get_labyrinth(labyrinth_file)
    print(labyrinth.shape)
    print(labyrinth)

    agent = play(labyrinth, alpha=alpha, gamma=gamma, eps=eps, n_trials=n_trials)
    camera, rewards, prop_succ = agent.td_sarsa()

    animation = camera.animate()
    animation.save('results/{}.gif'.format(labyrinth_file[5:-4]))
    plt.close()

    plt.title("Rewards v/s Episodes")
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('plots/{}_rewards.png'.format(labyrinth_file[5:-4]))
    plt.close()

    plt.title("Proportion of success v/s Number of trials")
    plt.plot(np.arange(len(prop_succ)), prop_succ)
    plt.xlabel('Episodes')
    plt.ylabel('Proportion of Success')
    plt.savefig('plots/{}_prop_successes.png'.format(labyrinth_file[5:-4]))
    plt.close()