#Python Code: Autonomous Robot Navigation with Deep Q-Learning
#shahab Baloochi.



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
import matplotlib.pyplot as plt
import time

# Environment parameters
GRID_SIZE = 5
ACTION_SPACE = 4  # Up, Down, Left, Right

# Deep Q-Network parameters
DISCOUNT = 0.95
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64
UPDATE_TARGET_EVERY = 5
EPISODES = 5000
EPSILON = 1  # exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

# Simulated robot environment
class Environment:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.agent_position = [0, 0]
        self.target_position = [self.grid_size - 1, self.grid_size - 1]
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        state[tuple(self.agent_position)] = 1
        state[tuple(self.target_position)] = 2
        return state.flatten()

    def step(self, action):
        if action == 0:  # up
            self.agent_position[1] = max(0, self.agent_position[1] - 1)
        elif action == 1:  # down
            self.agent_position[1] = min(self.grid_size - 1, self.agent_position[1] + 1)
        elif action == 2:  # left
            self.agent_position[0] = max(0, self.agent_position[0] - 1)
        elif action == 3:  # right
            self.agent_position[0] = min(self.grid_size - 1, self.agent_position[0] + 1)

        reward = -1
        done = False
        if self.agent_position == self.target_position:
            reward = 100
            done = True
        return self.get_state(), reward, done

# DQN model
def create_model(input_shape, action_space):
    model = Sequential()
    model.add(Dense(24, input_shape=input_shape, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    return model

# Experience replay memory
class ReplayMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size

# Visualization
def plot_grid(state, episode=None):
    grid = state.reshape((GRID_SIZE, GRID_SIZE))
    plt.imshow(grid, cmap='cool', interpolation='nearest')
    plt.title(f"Episode {episode}")
    plt.show()
    time.sleep(0.1)

# Main training loop
def train_agent():
    env = Environment(GRID_SIZE)
    model = create_model((GRID_SIZE**2,), ACTION_SPACE)
    target_model = create_model((GRID_SIZE**2,), ACTION_SPACE)
    target_model.set_weights(model.get_weights())

    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    target_update_counter = 0

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            if np.random.random() > EPSILON:
                action = np.argmax(model.predict(state.reshape(-1, *state.shape))[0])
            else:
                action = np.random.randint(0, ACTION_SPACE)

            next_state, reward, done = env.step(action)
            replay_memory.add((state, action, reward, next_state, done))
            state = next_state

            if replay_memory.can_sample(MINIBATCH_SIZE):
                minibatch = replay_memory.sample(MINIBATCH_SIZE)
                states = np.array([experience[0] for experience in minibatch])
                actions = np.array([experience[1] for experience in minibatch])
                rewards = np.array([experience[2] for experience in minibatch])
                next_states = np.array([experience[3] for experience in minibatch])
                dones = np.array([experience[4] for experience in minibatch])

                q_values = model.predict(states)
                q_values_next = target_model.predict(next_states)

                for i in range(MINIBATCH_SIZE):
                    if dones[i]:
                        q_values[i][actions[i]] = rewards[i]
                    else:
                        q_values[i][actions[i]] = rewards[i] + DISCOUNT * np.max(q_values_next[i])

                model.fit(states, q_values, batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

            target_update_counter += 1
            if target_update_counter >= UPDATE_TARGET_EVERY:
                target_model.set_weights(model.get_weights())
                target_update_counter = 0

        if EPSILON > MIN_EPSILON:
            global EPSILON
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)

        if episode % 100 == 0:
            print(f"Episode {episode}, EPSILON {EPSILON}")

    model.save("dqn_robot_navigation.h5")

# Test the trained agent
def test_agent(model, episodes=5):
    env = Environment(GRID_SIZE)
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = np.argmax(model.predict(state.reshape(-1, *state.shape))[0])
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
            plot_grid(state, episode)
            if done:
                print(f"Episode {episode} finished in {steps} steps.")

# Load a trained model
def load_model(filename):
    return tf.keras.models.load_model(filename)

# Run training
train_agent()

# After training, you can load the model and test it
# model = load_model("dqn_robot_navigation.h5")
# test_agent(model, episodes=5)
