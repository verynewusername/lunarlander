import random                       # Random
import torch                        # PyTorch
import numpy as np                  # NumPy
import torch.optim as optim         # PyTorch Optimizer Module
import torch.nn.functional as F     # PyTorch Functional Module

from qNetwork import QNetwork
from replayBuffer import ReplayBuffer

# ============== DQNAgent ============== #
class DQNAgent:

    # Variables to control the learning process
    timestep = 0    # Counter for the number of time steps taken in the environment
    batch_size = 64 # Size of the mini-batch used for training the neural network
    gamma = 0.99    # Discount factor for future rewards in the Q-learning update
    tau = 1e-3      # Soft update parameter for updating the fixed Q-network
    learning_rate = 0.0001  # Learning rate for the Adam optimizer
    update_every = 4  # Frequency of updating the Q-network

    def __init__(self, state_size, action_size, seed, bufferSize = 100000, batchSize = 64):
        self.seed = random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        # Q-network and fixed Q-network for learning and target Q-values
        self.q_network = QNetwork(state_size, action_size, seed)
        self.fixed_network = QNetwork(state_size, action_size, seed)
        # Adam optimizer for updating the Q-network
        self.optimizer = optim.Adam(self.q_network.parameters())
        # Replay buffer for storing experiences
        self.memory = ReplayBuffer(bufferSize, batchSize, seed)
        

    def step(self, state, action, reward, next_state, done):
        # Append the experience to the replay buffer
        self.memory.append(state, action, reward, next_state, done)
        self.timestep += 1
        # Perform a learning update every 'update_every' time steps
        if self.timestep % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Compute the Q-values for the next states using the fixed Q-network
        action_values = self.fixed_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # Compute the target Q-values for the Q-learning update
        Q_target = rewards + (self.gamma * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        
        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()

        # Backpropagate the gradients and perform a gradient descent step
        loss.backward()
        self.optimizer.step()
        
        # Update the fixed Q-network with soft updates
        self.update_fixed_network(self.q_network, self.fixed_network)
        
    def update_fixed_network(self, q_network, fixed_network):
        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(self.tau * source_parameters.data + (1.0 - self.tau) * target_parameters.data)
        
    def act(self, state, epsilon=0.0):
        rnd = random.random()
        # Explore with probability epsilon, otherwise exploit the learned policy
        if rnd < epsilon:
            return np.random.randint(self.action_size)
        else:
            # Evaluate the Q-network for the given state and choose the action with the highest Q-value
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.q_network.eval()

            with torch.no_grad():
                action_values = self.q_network(state)

            self.q_network.train()
            action = np.argmax(action_values.data.numpy())
            return action
        
    def checkpoint(self, filename):
        # Save the state_dict of the Q-network to the specified file
        torch.save(self.q_network.state_dict(), filename)
