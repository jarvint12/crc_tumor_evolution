import torch
import torch.nn as nn
import torch.nn.functional as F


class base_cnn(nn.Module):
    def __init__(self, many_classes):
        super(base_cnn, self).__init__()
        # YOUR CODE HERE
        self.many_classes=many_classes
        self.dropout=nn.Dropout(0.3) #changed from 0.2->0.3 1.6.2022
        self.m = nn.MaxPool1d(2, stride=2)
        self.conv1 = nn.Conv2d(1, 6, (4,3))
        self.norm1=nn.BatchNorm1d(6)
        self.conv2 = nn.Conv1d(6, 12, 3, padding=1)
        self.norm2=nn.BatchNorm1d(12)
        self.conv3 = nn.Conv1d(12, 12, 3, padding=1)
        self.norm3=nn.BatchNorm1d(12)
        self.conv4 = nn.Conv1d(12, 15, 3, padding=1)
        self.norm4=nn.BatchNorm1d(30)
        self.conv1_2 = nn.Conv2d(1, 15, (4,33))
        self.norm1_2=nn.BatchNorm2d(15)
        self.fc1=nn.Linear(30,200)#150) 
        self.bn1 = nn.BatchNorm1d(200)
        self.fc2=nn.Linear(200,120)
        self.bn2 = nn.BatchNorm1d(120)
        self.fc3=nn.Linear(120,8)
        self.fc4=nn.Linear(120,1)
        self.sigmoid=nn.Sigmoid()
        

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, 4, 33): Input sequences.
        
        Returns:
          y of shape (batch_size, 8): Outputs of the network.
        """
        y=self.conv1(x) #(batch_size, 6, 1, 31)
        y=self.dropout(y)
        y=F.relu(y)
        y=y.squeeze(dim=2) #(batch_size, 6, 31)
        y=self.m(y) #(batch_size, 6, 15)
        y=self.norm1(y)
        
        y=self.conv2(y) #(batch_size, 12, 15)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.m(y) #(batch_size, 12, 7)
        y=self.norm2(y)
        
        y=self.conv3(y) #(batch_size, 12, 7)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.m(y) #(batch_size, 12, 3)
        y=self.norm3(y)
        
        y=self.conv4(y) #(batch_size, 15, 3)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.m(y) #(batch_size, 15, 2)
        
        y2=self.conv1_2(x) #(batch_size, 15, 1, 1)
        y2=self.dropout(y2)
        y2=F.relu(y2)
        y2=y2.squeeze(dim=2) #(batch_size, 15, 1)
        
        y=torch.cat((y,y2), dim=2) #(batch_size, 15, 3)
        y = y.view(-1, self.num_flat_features(y)) #(batch_size, 45)
        
        y=self.norm4(y)
        
        y=self.fc1(y)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.bn1(y)
        
        y=self.fc2(y)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.bn2(y)
        if self.many_classes:
            y=self.fc3(y)
        return y
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features