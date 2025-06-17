import random                       # Random
import torch                        # PyTorch
import numpy as np                  # NumPy
import torch.optim as optim         # PyTorch Optimizer Module
import torch.nn.functional as F     # PyTorch Functional Module

from linearQ import linearQ
from replayBuffer import ReplayBuffer

def discretize_state(state, bins=1000):
    # Define bins for each dimension based on observed value ranges
    # Adjust these ranges based on your observations of the state values
    bins_list = [
        np.linspace(-1, 1, bins),  # Horizontal position
        np.linspace(-1.5, 1.5, bins),  # Vertical position
        np.linspace(-1, 1, bins),  # Horizontal velocity
        np.linspace(-1.5, 1.5, bins),  # Vertical velocity
        np.linspace(-np.pi, np.pi, bins),  # Angle
        np.linspace(-2, 2, bins)  # Angular velocity
    ]
    
    # Discretize each continuous state dimension
    discretized = []
    for value, bin_edges in zip(state[:6], bins_list):
        # np.digitize returns the index of the bin to which the value belongs
        # We subtract 1 to have 0-based indexing
        discretized_index = np.digitize(value, bin_edges) - 1
        discretized.append(discretized_index)
    
    # For the last two binary dimensions, simply append them as is
    discretized.extend(state[6:])
    
    return np.array(discretized)

# ============== Linear Agent ============== #
class LinearAgent:

    # Variables to control the learning process
    timestep = 0
    batch_size = 64
    gamma = 0.99
    tau = 1e-3
    learning_rate = 0.0001
    update_every = 4

    def __init__(self, state_size, action_size, seed, bufferSize = 100000, batchSize = 64):
        self.seed = random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.q_network = linearQ(state_size, action_size, seed)
        self.fixed_network = linearQ(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.q_network.parameters())

        self.memory = ReplayBuffer(bufferSize, batchSize, seed)
        

    def step(self, state, action, reward, next_state, done):
        self.memory.append(discretize_state(state), action, reward, next_state, done)
        # self.memory.append(state, action, reward, next_state, done)
        self.timestep += 1
        if self.timestep % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                sampled_experiences = self.memory.sample()
                self.learn(sampled_experiences)
        
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        action_values = self.fixed_network(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)
        
        Q_target = rewards + (self.gamma * max_action_values * (1 - dones))
        Q_expected = self.q_network(states).gather(1, actions)
        
        # Calculate loss
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()
        
        # Update fixed weights
        self.update_fixed_network(self.q_network, self.fixed_network)
        
    def update_fixed_network(self, q_network, fixed_network):
        for source_parameters, target_parameters in zip(q_network.parameters(), fixed_network.parameters()):
            target_parameters.data.copy_(self.tau * source_parameters.data + (1.0 - self.tau) * target_parameters.data)
        
    def act(self, state, epsilon=0.0):
        state = discretize_state(state)
        rnd = random.random()
        if rnd < epsilon:
            return np.random.randint(self.action_size)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.q_network.eval()

            with torch.no_grad():
                action_values = self.q_network(state)

            self.q_network.train()
            action = np.argmax(action_values.data.numpy())
            return action
        
    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)
