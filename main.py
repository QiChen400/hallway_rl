from __future__ import print_function
import pandas as pd
import os, sys, time, datetime, json, random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
from keras.models import model_from_json


class q_maze():
    def __init__(self, maze, start = (4, 5)):
        self.valid_actions = [0, 1, 2, 3]  # 0: up; 1: down; 2: left; 3: right
        self._maze = np.array(maze)
        self._state = (start, 'start')
        self._visited = {start}
        self._obs_cells = {(r, c) for r in range(self._maze.shape[0]) for c in range(self._maze.shape[1]) if
                            self._maze[r, c] == 1.0}

        self._free_cells = {(r, c) for r in range(self._maze.shape[0]) for c in range(self._maze.shape[1]) if
                            self._maze[r, c] == 0.0}
        self.reset()

    def reset(self):
        self.maze = np.copy(self._maze)
        self.state = self._state
        self.visited = {self.state[0]}

        self.covered = {}

    def get_new_state(self, action):
        nrows, ncols = self.maze.shape
        pos, mode = self.state
        row, col = pos
        if action == left and col > 0:
            col -= 1
        elif action == right and col < ncols-1:
            col += 1
        elif action == up and row > 0:
            row -= 1
        elif action == down and row < nrows-1:
            row += 1
        if (row, col) in self._obs_cells:
            return self.state
        return ((row, col), 'going')

    def act(self, action):
        old_state = self.state
        new_state = self.get_new_state(action)
        self.state = new_state

        if new_state == old_state:
            reward = self.get_reward(old_state, new_state, None, None)
            status = 'going'
        else:
            visited_num, visited_num_updated = self.update_maze()
            reward = self.get_reward(old_state, new_state, visited_num, visited_num_updated)
            status = self.game_status(visited_num_updated)
        envstate = self.observe()
        return envstate, reward, status

    def get_reward(self, old_state, new_state, visited_num, visited_num_updated):
        if new_state == old_state:
            return -5
        if visited_num_updated >= 0.9*len(self._free_cells):
            return 100
        return -1 + 0.1*(visited_num_updated-visited_num)

    def update_maze(self):
        nrows, ncols = self.maze.shape
        r_cur, c_cur = self.state[0]
        self.maze[r_cur, c_cur] = 2

        visited_num = len(self.visited)
        for r_nearby in range(r_cur-4, r_cur+5):
            for c_nearby in range(c_cur-4, c_cur+5):
                if 0 <= r_nearby and r_nearby < nrows and 0 <= c_nearby and c_nearby < ncols:
                    if self.maze[r_nearby, c_nearby] == 0:
                        self.maze[r_nearby, c_nearby] = 3
                        self.visited.add((r_nearby, c_nearby))
        visited_num_updated = len(self.visited)
        return visited_num, visited_num_updated

    def game_status(self, visited_num_updated):
        if visited_num_updated >= 0.9*len(self._free_cells):
            return 'win'
        return 'going'

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        nrows, ncols = self.maze.shape
        canvas = np.zeros((nrows, ncols))
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if self.maze[r, c] == 0:
                    canvas[r, c] = 0
                elif self.maze[r, c] == 1:
                    canvas[r, c] = 1
                elif self.maze[r, c] == 2:
                    canvas[r, c] = 0.7
                elif self.maze[r, c] == 3:
                    canvas[r, c] = 0.4
        # may need to consider since it is discrete data; also may need current state information as different color

        # # plotting
        # plt.grid('on')
        # ax = plt.gca()
        # ax.set_xticks(np.arange(0.5, nrows, 1))
        # ax.set_yticks(np.arange(0.5, ncols, 1))
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # plt.imshow(canvas, interpolation='none', cmap='gray')
        # plt.show()
        return canvas


class Experience():
    def __init__(self, model, max_memory = 100, discount = 0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    # def predict(self, envstate):
    #     reshaped_envstate = envstate.reshape(1, 10, 10, 1)
    #     # print(envstate.shape)
    #     return self.model.predict(reshaped_envstate)[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

def train(model, maze, **opt):
    n_epoch = opt.get('n_epoch', 3)
    data_size = opt.get('data_size', 32)
    max_memory = opt.get('max_memory', 100)
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    qmaze = q_maze(date_maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)
    win_history = []  # history of win/lose game
    hsize = qmaze.maze.size // 2  # history window size
    win_rate = 0.0

    for epoch in range(n_epoch):
        loss = 0
        qmaze.reset()
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()

        n_episodes = 0
        while not game_over:
            prev_envstate = envstate
            # Get next action
            # if np.random.rand() < epsilon and epoch <= 30:
            if np.random.rand() < epsilon:
                action = random.choice(qmaze.valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            # print(inputs.shape)
            # inputs = inputs.reshape((-1,10,10,1))
            # print(inputs)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} "
        print(template.format(epoch, n_epoch - 1, loss, n_episodes, sum(win_history), win_rate))

    # Save trained model weights and architecture, this will be used by the visualization code
    path = '/Users/qchen2/PycharmProjects/hallway_rl/'
    h5file = path + name + ".h5"
    json_file = path + name + ".json"
    model.save_weights(h5file, overwrite=True)
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile)
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print('files: %s, %s' % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, data: %d, time: %s" % (epoch, max_memory, data_size, t))
    return seconds

def test(maze):
    with open("/Users/qchen2/PycharmProjects/hallway_rl/model.json", "r") as jfile:
        model = model_from_json(json.load(jfile))
    model.load_weights("/Users/qchen2/PycharmProjects/hallway_rl/model.h5")
    model.compile("adam", "mse")

    # Define environment, game
    qmaze = q_maze(maze)
    c = 0
    for e in range(1):
        qmaze.reset()
        game_over = False

        # Initialize experience replay object
        experience = Experience(model, max_memory=100)

        c += 1
        while not game_over:
            # get initial envstate (1d flattened canvas)
            envstate = qmaze.observe()

            plt.imshow(envstate.reshape(10, -1),
                       interpolation='none', cmap='gray')
            plt.savefig("/Users/qchen2/PycharmProjects/hallway_rl/image/%03d.png" % c)

            # get next action
            action = np.argmax(experience.predict(envstate))

            # Apply action, get reward and new envstate
            new_envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                game_over = True
            elif game_status == 'lose':
                game_over = True
            else:
                game_over = False

            c += 1


def build_model(maze, lr=0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')
    return model



# some hyper-parameters
left = 0
right = 1
up = 2
down = 3

# Actions dictionary
actions_dict = {
    left: 'left',
    up: 'up',
    right: 'right',
    down: 'down',
}

# # data source
# date_maze = np.array([
#     [1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.],
#     [1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.],
#     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.],
#     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#     [0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
#     [0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
#     [1.,  1.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.],
#     [1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#     [1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
#     [1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.]
# ])

outline = np.array([[0, 0], [150, 0], [150, 100], [0, 100], [0, 0]])/5
out_obs = np.array([[[0, 70], [50, 70], [50, 100], [0, 100], [0, 70]],
                    [[130, 80], [150, 80], [150, 100], [130, 100], [130, 80]],
                    [[0, 0], [20, 0], [20, 10], [0, 10], [0, 0]],
                    [[140, 0], [150, 0], [150, 20], [140, 20], [140, 0]]])/5
core = np.array([[20, 15], [60, 15], [60, 60], [20, 60], [20, 15]])/5
date_maze = np.zeros((int(outline[2][0]), int(outline[2][1])))
for obs in out_obs:
    for i in range(int(obs[0][0]), int(obs[2][0])):
        for j in range(int(obs[0][1]), int(obs[2][1])):
            date_maze[i][j] = 1.0

for i in range(int(core[0][0]), int(core[2][0])):
    for j in range(int(core[0][1]), int(core[2][1])):
        date_maze[i][j] = 1.0
print(date_maze)

# Exploration factor
epsilon = 0.1


model = build_model(date_maze)
train(model, date_maze, n_epoch=100, max_memory=100, data_size=32)
test(date_maze)