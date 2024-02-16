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
        d_model=84
        self.embedding = nn.Embedding(5, d_model)
        self.dropout=nn.Dropout(0.3)
        self.m = nn.MaxPool1d(2, stride=2)
        self.conv1 = nn.Conv2d(1, 6, (84,1))
        self.norm1=nn.BatchNorm1d(6)
        self.conv2 = nn.Conv1d(6, 12, 3)
        self.conv1_2 = nn.Conv2d(1, 12, (84,33))
        self.norm3=nn.BatchNorm1d(404)
        self.fc1=nn.Linear(404,200)
        self.norm4=nn.BatchNorm1d(200)
        self.fc2=nn.Linear(200,120)
        self.norm5=nn.BatchNorm1d(120)
        self.fc3=nn.Linear(120,8)
        

    def forward(self, x, mut_landscape):
        """
        Args:
          x of shape (batch_size, 33): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        x = self.embedding(x.to(torch.long)) #(batch_size, 33, 84)
        x = x.permute(0, 2, 1) #(batch_size, 84, 33)
        x = x.unsqueeze(1) #(batch_size, 1, 84, 33) -> (batch_size, C, H, W)
        y=self.conv1(x)
        y=self.dropout(y)
        y=F.relu(y) # (batch_size, 6, 1, 33)
        y=y.squeeze(dim=2) # (batch_size, 6, 33)
        y=self.norm1(y)
        
        y=self.conv2(y)
        y=self.dropout(y)
        y=F.relu(y) # (batch_size, 12, 31)
        
        y2=self.conv1_2(x) #(batch_size, 1, 84, 33) -> #(batch_size, 12, 1, 1)
        y2=self.dropout(y2)
        y2=F.relu(y2).squeeze(dim=2) # (batch_size, 21, 4, 33) -> (batch_size, 12, 1)
        y=torch.cat((y,y2), dim=2) # (batch_size, 12, 31+1)
        
        y = y.view(-1, self.num_flat_features(y)) # (batch_size, 384)
        y=torch.cat((y,mut_landscape), dim=1) # (batch_size, 384+20)
        y=self.norm3(y)
        
        y=self.fc1(y)
        y=self.dropout(y)
        y=F.relu(y) # (batch_size, 200)
        y=self.norm4(y)
        
        y=self.fc2(y)
        y=self.dropout(y)
        y=F.relu(y) # (batch_size, 120)
        y=self.norm5(y)
        
        y=self.fc3(y) # (batch_size, 8)
        return y
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features