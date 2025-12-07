import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.input_dim=input_dim
        self.n_actions = n_actions
        #define structure
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)
        
        # For tracking losses
        self.loss_history = []

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # add batch dimension if missing
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
    # Method to record loss during training
    def record_loss(self, loss):
        self.last_loss = loss.item()
        self.loss_history.append(self.last_loss)
    #this will modify the current model and return a mutated version for evolution
    def clone_with_mutation(self, noise_scale=0.02):
        new_model = DQN(self.input_dim, self.n_actions)
        new_model.load_state_dict(self.state_dict())

        with torch.no_grad():
            for p in new_model.parameters():
                p.data += noise_scale * torch.randn_like(p)
        return new_model
#store game transitions for training purposes
class ReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.stack(s2), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)