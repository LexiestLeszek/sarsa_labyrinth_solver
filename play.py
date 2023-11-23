import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from celluloid import Camera


def get_states(m):
    TS, IS, FS = ['f', 'nv', 'oob'], [], []

    for i in range(0, len(m)):
        for j in range(0, len(m[i])):
            if m[i][j] == 's':
                IS.append((i, j))
            if m[i][j] == 'f':
                FS.append((i, j))

    return TS, IS, FS


def get_state_space(m):
    SS = []
    h, w = m.shape
    for i in range(0, h):
        for j in range(0, w):
            SS.append((i, j))
    return SS


def init_q(ss, acs):
    Q = {}
    for s in ss:
        for a1 in acs:
            for a2 in acs:
                Q[s, (a1, a2)] = 0
    for a1 in acs:
        for a2 in acs:
            Q[(-1, -1), (a1, a2)] = 0  # non-valid states are denoted by (-1, -1)
    return Q


class play:
    def __init__(self, maze, alpha, gamma, eps, train=True, eps_decay=True, n_trials=10000):
        self.maze = maze
        self.ml = maze.shape[1]
        self.mw = maze.shape[0]
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.train = train
        self.eps_decay = eps_decay
        self.n_trials = n_trials

        self.TS, self.IS, self.FS = get_states(self.maze)
        self.inc_action_space = [-1, 0, 1]
        self.action_space = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        self.state_space = get_state_space(self.maze)

        self.action_value_function = init_q(self.state_space, self.action_space)
        self.gif = []
        self.rewards = []
        self.n_succ = []

        self.fig, self.ax = plt.subplots()
        self.camera = Camera(self.fig)

    def check_move(self, prev_state, curr_state):
        """
        Takes in previous and current states and checks if the move is valid or not
        :param prev_state: previous state
        :param curr_state: current state
        :return: class of move = {'v', 'nv', 'oob', 'f'}
        """
        x1, y1 = prev_state[1], prev_state[0]
        x2, y2 = curr_state[1], curr_state[0]
        move = 'v'
        # checking if the move is out of bounds
        if not (-1 < x1 < self.ml and -1 < x2 < self.ml and -1 < y1 < self.mw and -1 < y2 < self.mw):
            move = 'oob'
            return move

        min_y, min_x = np.min([y1, y2]), np.min([x1, x2])
        max_y, max_x = np.max([y1, y2]), np.max([x1, x2])
        sub_mat = self.maze[min_y:max_y + 1, :]
        sub_mat = sub_mat[:, min_x:max_x + 1]

        # checking if there are any barriers present in the sub matrix
        if 'x' in sub_mat:
            move = 'nv'
            return move
        # Checking if the move has reached the finish point
        if 'f' in sub_mat:
            move = 'f'
            return move
        return move

    def take_action(self, prev_action, prev_state):
        """
        epsilon greedily returns new action
        :param prev_state: previous state
        :param prev_action: previous action
        :return: new action
        """
        if np.random.random() < self.eps:
            while True:
                vel_action = tuple(random.sample(self.inc_action_space, 2))
                if (prev_action[0] + vel_action[0]) == 0 and (prev_action[1] + vel_action[1]) == 0:
                    continue
                elif abs(prev_action[0] + vel_action[0]) >= 5 or abs(prev_action[1] + vel_action[1]) >= 5:
                    continue
                else:
                    return vel_action
        else:
            while True:
                tup = tuple(random.sample(self.inc_action_space, 2))
                if prev_action[0] + tup[0] == 0 and prev_action[1] + tup[1] == 0:
                    # print('here')
                    continue
                elif abs(prev_action[0] + tup[0]) >= 5 or abs(prev_action[1] + tup[1]) >= 5:
                    # print('here_again')
                    continue
                else:
                    break
            greedy_action_value = self.action_value_function[prev_state, (prev_action[0] + tup[0], prev_action[1] + tup[1])]
            for a1 in self.inc_action_space:
                for a2 in self.inc_action_space:
                    if (prev_action[0] + a1) == 0 and (prev_action[1] + a2) == 0:
                        continue
                    elif abs(prev_action[0] + a1) >= 5 or abs(prev_action[1] + a2) >= 5:
                        continue
                    else:
                        if greedy_action_value < self.action_value_function[prev_state, (prev_action[0] + a1, prev_action[1]
                                                                                                        + a2)]:
                            greedy_action_value = self.action_value_function[prev_state, (prev_action[0] + a1, prev_action[1]
                                                                                    + a2)]
                            tup = (a1, a2)
            return tup

    def reward_function(self, prev_state, curr_state):
        """
        takes in previous and current states and grants rewards accordingly for finish, out of bounds non-valid and
        valid moves
        :param prev_state: previous state
        :param curr_state: current state
        :return: rewards obtained
        """
        move = self.check_move(prev_state, curr_state)
        r = 0
        if move == 'f':
            r = 10
        elif move == 'oob':
            r = -20
        elif move == 'nv':
            r = -10
        elif move == 'v':
            r = -1
        return r

    def td_sarsa(self):
        count = 0
        gif_counter = 0
        while count <= self.n_trials:
            curr_state = random.sample(self.IS, 1)[0]
            v = (0, 0)
            action = self.take_action(v, curr_state)
            v = (v[0] + action[0], v[1] + action[1])
            rew = 0
            states = [curr_state]
            while True:
                new_state = (curr_state[0] + v[0], curr_state[1] + v[1])
                move = self.check_move(curr_state, new_state)
                if move == 'oob':
                    new_state = (-1, -1)
                new_rew = self.reward_function(curr_state, new_state)
                new_action = self.take_action(v, new_state)
                v1 = (v[0] + new_action[0], v[1] + new_action[1])
                rew = rew + new_rew
                self.action_value_function[curr_state, v] = (self.action_value_function[curr_state, v]
                                                             + self.alpha * (new_rew + self.gamma
                                                                             * self.action_value_function[new_state, v1]
                                                                             - self.action_value_function[
                                                                                 curr_state, v]))
                curr_state = new_state
                v = v1
                states.append(curr_state)
                self.rewards.append(rew)
                if move in self.TS:
                    if move == 'f':
                        count += 1
                    gif_counter += 1
                    if not gif_counter % 1000:
                        self.take_snap(states)
                    if not gif_counter % 10000:
                        print('Num trials = {} \t Num successes = {} \t Prop success = {}'.format(gif_counter, count,
                                                                                                  round(count/gif_counter, 3)))
                    self.n_succ.append(round(count/gif_counter, 3))
                    break
            if self.train and self.eps_decay:
                self.eps = self.eps - count / 10000 if self.eps > 0.01 else 0.01

        return self.camera, self.rewards, self.n_succ

    def take_snap(self, states):
        m1 = copy.deepcopy(self.maze)
        nrows, ncols = self.mw, self.ml
        image = np.zeros(nrows * ncols)
        image = image.reshape((nrows, ncols))
        for s in states:
            if s != (-1, -1):
                m1[s] = 'g'
        for i in range(0, nrows):
            for j in range(0, ncols):
                if m1[i][j] == 's':
                    image[i][j] = 0
                if m1[i][j] == '0':
                    image[i][j] = 1
                if m1[i][j] == 'x':
                    image[i][j] = 2
                if m1[i][j] == 'f':
                    image[i][j] = 3

        self.gif.append(image)
        self.ax.imshow(image)
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        self.ax.set_xticks(np.arange(-.5, ncols, 1))
        self.ax.set_yticks(np.arange(-.5, nrows, 1))
        self.camera.snap()
