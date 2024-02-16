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
        self.fc1=nn.Linear(24,120)#150) 
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2=nn.Linear(120,8)
        self.sigmoid=nn.Sigmoid()
        

    def forward(self, sequences, mut_landscape):
        """
        Args:
          sequences of shape (batch_size, 33): Input sequences.
          mut_landscape of shape (batch_size, 20): Input sequences.
        
        Returns:
          y of shape (batch_size, 8): Outputs of the network.
        """
        
        x = self.create_one_hot_input(next(self.parameters()).device, sequences)
        x=torch.cat((x,mut_landscape), dim=1) # (batch_size, 4+20)
        y=self.fc1(x)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.bn1(y)
        
        y=self.fc2(y)
        return y


    def create_one_hot_input(self, device, sequences):
      """Converts numbered sequence to one-hot based on the middle base
      
      Args:
      sequences of shape (batch_size, 33): Input sequences.
      
      Returns:
      sequences of shape (batch_size, 4): Target sequence one-hot encoded."""

      # Define the conditions
      condition_1 = (sequences[:, 16] == 1)
      condition_2 = (sequences[:, 16] == 2)
      condition_3 = (sequences[:, 16] == 3)
      condition_4 = (sequences[:, 16] == 4)
      
      one_hot = torch.zeros((sequences.shape[0], 4), device = device)
      one_hot[condition_1, 0] = 1
      one_hot[condition_2, 1] = 1
      one_hot[condition_3, 2] = 1
      one_hot[condition_4, 3] = 1
      return one_hot
