import torch                        # PyTorch    
import torch.nn as nn               # PyTorch Neural Network Module
import torch.nn.functional as F     # PyTorch Functional Module

# ============== QNetwork ============== #
class QNetwork(nn.Module):
    # The best Performing Model had 4 Layers Therefore we leave 4 layers as default
    def __init__(self, state_size, action_size, seed, hidden_layer_size=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        # self.fc4 = nn.Linear(hidden_layer_size, hidden_layer_size)
        # self.fc5 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc5 = nn.Linear(hidden_layer_size, action_size)  
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.fc5(x)
        return x
    
    def get_active_layer_count(self):
        active_layers = sum(layer is not None for layer in self.children())
        return active_layers
    