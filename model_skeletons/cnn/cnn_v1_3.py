import torch.nn as nn
import torch.nn.functional as F

model_version="cnn_v1_3"

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # YOUR CODE HERE
        self.dropout=nn.Dropout(0.3)
        self.conv1 = nn.Conv2d(21, 6, (4,1))
        self.norm1=nn.BatchNorm1d(6)
        self.conv2 = nn.Conv1d(6, 12, 3)
        self.norm2=nn.BatchNorm1d(12)
        self.conv3 = nn.Conv1d(12, 15, 3)
        self.norm3=nn.BatchNorm1d(15)
        self.fc1=nn.Linear(435,250)#150)
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
        y=self.conv1(x) # (batch_size, 25, 1, 33)
        y=self.dropout(y)
        y=F.relu(y)
        y=y.squeeze(dim=2) # (batch_size, 6, 33)
        y=self.norm1(y) #Norm over C dimension of (N, C, L)
           
        y=self.conv2(y) # (batch_size, 12, 31)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.norm2(y)
        
        y=self.conv3(y) # (batch_size, 15, 29)
        y=self.dropout(y)
        y=F.relu(y)
        y=self.norm3(y)
        y = y.view(-1, self.num_flat_features(y)) # (batch_size, 435)
        
        y=self.fc1(y) # (batch_size, 250)
        y=F.relu(y)
        y=self.dropout(y)
        
        y=self.fc2(y)  # (batch_size, 150)
        y=F.relu(y)
        y=self.dropout(y)
        
        y=self.fc3(y)  # (batch_size, 8)
        return y
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features