import torch
import torch.nn as nn
import torch.nn.functional as F
import os

model_version=os.path.basename(__file__).split(".")[0]

class base_fc(nn.Module):
    def __init__(self):
        super(base_fc, self).__init__()
        # YOUR CODE HERE
        self.dropout=nn.Dropout(0.3) #changed from 0.2->0.3 1.6.2022
        self.fc1=nn.Linear(84,300)#150) 
        self.bn1 = nn.BatchNorm1d(300)
        self.fc2=nn.Linear(300,200)
        self.bn2 = nn.BatchNorm1d(200)
        self.fc3=nn.Linear(200,120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc4=nn.Linear(120,8)
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
        y=self.dropout(y)
        y=F.relu(y)
        y=self.bn2(y)
        
        y=self.fc3(y)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.bn3(y)
        
        y=self.fc4(y)
        
        return y