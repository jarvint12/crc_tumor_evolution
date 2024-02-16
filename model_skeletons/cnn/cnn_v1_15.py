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
        self.norm0=nn.BatchNorm2d(21)
        self.conv1 = nn.Conv2d(21, 60, (4,3))
        self.norm1=nn.BatchNorm1d(60)
        self.conv2 = nn.Conv1d(60, 60, 3)
        self.norm2=nn.BatchNorm1d(60)
        self.conv3 = nn.Conv1d(60, 60, 3)
        self.norm3=nn.BatchNorm1d(60)
        self.conv1_2 = nn.Conv2d(21, 60, (4,33))
        self.fc1=nn.Linear(180,200)#150) 
        self.fc2=nn.Linear(200,120)
        self.fc3=nn.Linear(120,8)
        

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 21, 4, 33): Input sequences.
        
        Returns:
          y of shape (batch_size, 8): Outputs of the network.
        """
        x=self.norm0(x)
        y=F.relu(self.conv1(x)) #(N, C, 1, L), where N is batch size, C is number of channels, L is length after CNN
        y=y.squeeze(dim=2) #(N, C, L)
        y=self.m(y)
        
        y=self.norm1(y) #Norm over C dimension of (N, C, L)
        y=F.relu(self.conv2(y))
        y=self.m(y)
        
        y=self.norm2(y)
        y=self.dropout(y)
        y=F.relu(self.conv3(y))
        y=self.m(y)
        y=self.dropout(y)
        
        y2=F.relu(self.conv1_2(x)).squeeze(dim=2) #(N, C, 1, 1) -> (N, C, 1)
        
        y=torch.cat((y,y2), dim=2) # (N, C, L+1)
        
        y=self.norm3(y)
        y = y.view(-1, self.num_flat_features(y)) #(N, C*(L+1))
        y=F.relu(self.fc1(y))
        y=self.dropout(y)
        
        y=F.relu(self.fc2(y))
        y=self.dropout(y)
        
        y=self.fc3(y)
        
        return y
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features