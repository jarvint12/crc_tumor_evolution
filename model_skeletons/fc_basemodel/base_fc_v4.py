import torch
import torch.nn as nn
import torch.nn.functional as F
import os

model_version=os.path.basename(__file__).split(".")[0]

class base_fc(nn.Module):
    def __init__(self):
        super(base_fc, self).__init__()
        # YOUR CODE HERE
        self.dropout=nn.Dropout(0.2) #changed from 0.2->0.3 1.6.2022
        self.fc1=nn.Linear(84,30)#150) 
        self.bn1 = nn.BatchNorm1d(30)
        self.fc2=nn.Linear(30,8)
        self.sigmoid=nn.Sigmoid()
        

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 84): Input sequences.
        
        Returns:
          y of shape (batch_size, 8): Outputs of the network.
        """
        
        
        y=self.fc1(x)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.bn1(y)
        
        y=self.fc2(y)
        return y