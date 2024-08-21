#different reinforcement learning method: SARSA (State-Action-Reward-State-Action)

# shahab baloochi


import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

# Environment parameters
GRID_SIZE = 5
ACTION_SPACE = 4  # Up, Down, Left, Right

# SARSA parameters
ALPHA = 0.1  # Learning rate
DISCOUNT = 0.95
EPISODES = 5000
EPSILON = 1  # exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.001

# Performance tracking
REWARDS = []
STEPS = []

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

# SARSA policy
def choose_action(state, q_table, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, ACTION_SPACE)
    else:
        return np.argmax(q_table[state])

# Visualization of the Q-table as a heatmap
def plot_q_table(q_table, episode=None):
    q_table_max = np.max(q_table, axis=1).reshape(GRID_SIZE, GRID_SIZE)
    sns.heatmap(q_table_max, annot=True, cmap='coolwarm', cbar=True)
    plt.title(f"Q-Table at Episode {episode}" if episode else "Q-Table")
    plt.show()

# Visualization of the agent's path
def plot_grid(state, episode=None, steps=None):
    grid = state.reshape((GRID_SIZE, GRID_SIZE))
    plt.imshow(grid, cmap='cool', interpolation='nearest')
    title = f"Episode {episode}" if episode is not None else "Agent Path"
    if steps is not None:
        title += f", Steps: {steps}"
    plt.title(title)
    plt.show()
    time.sleep(0.1)

# Main training loop using SARSA
def train_agent():
    global EPSILON
    env = Environment(GRID_SIZE)
    q_table = np.zeros((GRID_SIZE**2, ACTION_SPACE))

    for episode in range(EPISODES):
        state = env.reset()
        state_index = np.argmax(state)
        action = choose_action(state_index, q_table, EPSILON)

        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            next_state, reward, done = env.step(action)
            next_state_index = np.argmax(next_state)
            next_action = choose_action(next_state_index, q_table, EPSILON)

            # SARSA update
            q_table[state_index][action] += ALPHA * (reward + DISCOUNT * q_table[next_state_index][next_action] - q_table[state_index][action])

            state_index = next_state_index
            action = next_action
            episode_reward += reward
            episode_steps += 1

        REWARDS.append(episode_reward)
        STEPS.append(episode_steps)

        if EPSILON > MIN_EPSILON:
            EPSILON *= EPSILON_DECAY
            EPSILON = max(MIN_EPSILON, EPSILON)

        if episode % 100 == 0:
            print(f"Episode {episode}, EPSILON {EPSILON}, Reward: {episode_reward}, Steps: {episode_steps}")
            plot_q_table(q_table, episode)

    np.save("sarsa_q_table.npy", q_table)

# Test the trained agent using the Q-table
def test_agent(q_table, episodes=5):
    env = Environment(GRID_SIZE)
    for episode in range(episodes):
        state = env.reset()
        state_index = np.argmax(state)
        done = False
        steps = 0
        while not done:
            action = np.argmax(q_table[state_index])
            next_state, reward, done = env.step(action)
            state_index = np.argmax(next_state)
            steps += 1
            plot_grid(state, episode, steps)
            if done:
                print(f"Episode {episode} finished in {steps} steps.")

# Plot training performance
def plot_training_performance(rewards, steps):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(rewards)
    axs[0].set_title('Rewards Over Time')
    axs[0].set_xlabel('Episodes')
    axs[0].set_ylabel('Total Reward')

    axs[1].plot(steps)
    axs[1].set_title('Steps Over Time')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Steps Taken')

    plt.tight_layout()
    plt.show()

# Run training
train_agent()

# Plot the performance after training
plot_training_performance(REWARDS, STEPS)

# After training, you can load the Q-table and test it
# q_table = np.load("sarsa_q_table.npy")
# test_agent(q_table, episodes=5)
