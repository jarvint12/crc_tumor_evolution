import torch.nn as nn
import torch.nn.functional as F

model_version="cnn_v1_2"

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # YOUR CODE HERE
        self.dropout=nn.Dropout(0.3)
        self.m = nn.MaxPool1d(2, stride=2)
        self.norm0=nn.BatchNorm2d(21)
        self.conv1 = nn.Conv2d(21, 6, (4,1))
        self.norm1=nn.BatchNorm1d(6)
        self.conv2 = nn.Conv1d(6, 12, 3)
        self.norm2=nn.BatchNorm1d(12)
        self.conv3 = nn.Conv1d(12, 15, 3)
        self.norm3=nn.BatchNorm1d(15)
        self.fc1=nn.Linear(30,250)#150)
        self.bn1 = nn.BatchNorm1d(250)
        self.fc2=nn.Linear(250,150)
        self.bn2 = nn.BatchNorm1d(150)
        self.fc3=nn.Linear(150,8)
        

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 21, 4, 33): Input sequences.
        
        Returns:
          y of shape (batch_size, 8): Outputs of the network.
        """
        y=self.conv1(x)
        y=self.dropout(y)
        y=F.relu(y) #(N, C, 1, L), where N is batch size, C is number of channels, L is length after CNN
        y=y.squeeze(dim=2) #(N, C, L)
        y=self.m(y)
        y=self.norm1(y) #Norm over C dimension of (N, C, L)
           
        y=self.conv2(y)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.m(y)
        y=self.norm2(y)
        
        y=self.conv3(y)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.m(y)
        y=self.norm3(y)
        y = y.view(-1, self.num_flat_features(y))
        
        y=self.fc1(y)
        y=F.relu(y)
        y=self.dropout(y)
        
        y=self.fc2(y)
        y=F.relu(y)
        y=self.dropout(y)
        
        y=self.fc3(y)
        return y
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features