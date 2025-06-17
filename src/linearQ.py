import torch                        # PyTorch    
import torch.nn as nn               # PyTorch Neural Network Module


# ============== linearQ ============== #
class linearQ(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(linearQ, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.Linear(state_size, action_size)
        
    def forward(self, state):
        return self.fc(state)
    