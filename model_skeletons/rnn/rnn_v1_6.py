import torch
import torch.nn as nn
import torch.nn.functional as F
import os

current_file_name = os.path.basename(__file__)
model_version=current_file_name.split('.')[0]

class rnn(nn.Module):
    def __init__(self, bidirectional, hidden_dim, n_layers):
        super(rnn, self).__init__()
        # YOUR CODE HERE
        input_size=84
        output_size=8
        self.norm0=nn.BatchNorm2d(1)
        self.bidirectional=bidirectional #True
        self.hidden_dim=hidden_dim #35
        self.n_layers = n_layers
        fc_input=33*(int(self.bidirectional)+1)*self.hidden_dim
        self.dropout=nn.Dropout(0.2)
        self.rnn = nn.RNN(input_size, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=self.bidirectional, 
                          nonlinearity='relu')
        self.fc1 = nn.Linear(fc_input, 500)
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, output_size)

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, 1, length (33), height (84)): Input sequences.
        
        Returns:
          y of shape (batch_size, 8): Outputs of the network.
        """
        x=torch.squeeze(x, dim=1) #(batch_size,L,H)
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden) #(N,L,D∗hidden_dim), where D = 2 if bidirectional=True otherwise 1
        #out = out.contiguous().view(-1, self.hidden_dim)
        out = out.contiguous().view(-1, self.num_flat_features(out)) #(N, L*D*hidden_dim)
        out=F.relu(self.fc1(out))
        out=self.dropout(out)
        out=F.relu(self.fc2(out))
        out=self.dropout(out)
        out=self.fc3(out)
        return out, hidden
    

    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden = torch.zeros((int(self.bidirectional)+1)*self.n_layers, batch_size, self.hidden_dim, device=device)
        return hidden
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features