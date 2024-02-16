import torch.nn as nn
import torch.nn.functional as F

model_version="cnn_v1_4"

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        # YOUR CODE HERE
        self.dropout=nn.Dropout(0.3)
        self.m = nn.MaxPool1d(2, stride=2)
        self.conv1 = nn.Conv2d(21, 25, (4,1))
        self.conv2 = nn.Conv1d(25, 12, 3)
        self.conv3 = nn.Conv1d(12, 15, 3)
        self.fc1=nn.Linear(435,200)
        self.fc2=nn.Linear(200,120)
        self.fc3=nn.Linear(120,8)
        
#RuntimeError: Given groups=1, weight of size [15, 12, 3], expected input[128, 25, 33] to have 12 channels, but got 25 channels instead
    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, 28, 28): Input images.
        
        Returns:
          y of shape (batch_size, 10): Outputs of the network.
        """
        y=self.conv1(x)
        y=self.dropout(y)
        y=F.relu(y)
        y=y.squeeze(dim=2)
        
        y=self.conv2(y)
        y=self.dropout(y)
        y=F.relu(y)
        
        y=self.conv3(y)
        y=self.dropout(y)
        y=F.relu(y)
        y = y.view(-1, self.num_flat_features(y))
        
        y=self.fc1(y)
        y=self.dropout(y)
        y=F.relu(y)
        
        y=self.fc2(y)
        y=self.dropout(y)
        y=F.relu(y)
        
        y=self.fc3(y)
        return y
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features