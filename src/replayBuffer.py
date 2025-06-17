import torch                        # PyTorch
from collections import deque       # Deque
import random                       # Random
import numpy as np                  # NumPy

# ============== ReplayBuffer ============== #
class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed = 42):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Torch expects the inputs to be tensors
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()     

        # Return a single random sample
        return (states, actions, rewards, next_states, dones) 

    def append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def __len__(self):
        return len(self.memory)