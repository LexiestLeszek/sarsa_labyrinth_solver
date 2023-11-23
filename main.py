import os
import numpy as np
import argparse


def get_maze(maze_file):
    maze = []
    with open(maze_file, 'r') as f:
        for line in f:
            maze.append(line)
    maze = np.array([list(line)[:-1] for line in maze])
    return maze



if __name__ == "__main__":
    maze_file = 'data/labyrinth1.txt'
    l = get_maze(maze_file)

    print(l)