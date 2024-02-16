import torch
import torch.nn as nn
import torch.nn.functional as F
import os

current_file_name = os.path.basename(__file__)
model_version=current_file_name.split('.')[0]

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # YOUR CODE HERE
        self.dropout=nn.Dropout(0.3)
        self.m = nn.MaxPool1d(2, stride=2)
        self.conv1 = nn.Conv2d(21, 6, (4,1))
        self.conv2 = nn.Conv1d(6, 12, 3)
        self.conv3 = nn.Conv1d(12, 15, 3)
        self.conv1_2 = nn.Conv2d(21, 15, (4,33))
        self.fc1=nn.Linear(450,200)
        self.fc2=nn.Linear(200,120)
        self.fc3=nn.Linear(120,8)
        

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 21, 4, 33): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        y=F.relu(self.conv1(x)) # (batch_size, 6, 1, 33)
        y=y.squeeze(dim=2) # (batch_size, 6, 33)
        y=F.relu(self.conv2(y)) # (batch_size, 12, 31)
        y=self.dropout(y)
        
        y=F.relu(self.conv3(y)) # (batch_size, 15, 29)
        y=self.dropout(y)
        
        y2=F.relu(self.conv1_2(x)).squeeze(dim=2) # (batch_size, 21, 4, 33) -> (batch_size, 15, 1)
        y=torch.cat((y,y2), dim=2) # (batch_size, 15, 30)
        #y=self.norm3(y)
        
        y = y.view(-1, self.num_flat_features(y)) # (batch_size, 450)
        
        y=F.relu(self.fc1(y)) # (batch_size, 200)
        y=self.dropout(y)
        
        y=F.relu(self.fc2(y)) # (batch_size, 120)
        y=self.dropout(y)
        
        y=self.fc3(y) # (batch_size, 8)
        return y
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features