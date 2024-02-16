import torch
import torch.nn as nn
import torch.nn.functional as F
import os

model_version=os.path.basename(__file__).split(".")[0]

class base_fc(nn.Module):
    def __init__(self):
        super(base_fc, self).__init__()
        # YOUR CODE HERE
        self.fc1=nn.Linear(84,8)#150) 
        

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 84): Input sequences.
        
        Returns:
          y of shape (batch_size, 8): Outputs of the network.
        """
        
        
        y=self.fc1(x)
        return y