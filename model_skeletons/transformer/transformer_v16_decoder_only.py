import torch
import torch.nn as nn
import os
import math

current_file_name = os.path.basename(__file__)
model_version=current_file_name.split('.')[0]

class WeightedMSELoss(nn.Module):
    def __init__(self, device, class_weights, classes, correct_label_index, indices_of_interest):
        super(WeightedMSELoss, self).__init__()
        self.weights=class_weights
        self.classes=classes
        self.num_classes=len(self.classes)
        self.correct_label_index=correct_label_index
        self.indices_of_interest = indices_of_interest
        self.bases= {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
        self.device=device
            
    def forward(self, sequences, targets, outputs):
        values_of_interest = sequences[:, 16, self.indices_of_interest]
        _, original_bases = torch.max(values_of_interest, dim=1)
        
        values_of_interest = targets[:, self.correct_label_index, self.indices_of_interest]
        _, new_bases = torch.max(values_of_interest, dim=1)
        correct_classes = [self.classes[self.bases[original.item()] + self.bases[new_.item()]] for original, new_ in zip(original_bases, new_bases)]

        # Calculate weights based on correct classes
        weights = torch.tensor([self.weights[class_] for class_ in correct_classes]).to(self.device)
        weights = weights.view(-1, 1, 1).expand_as(outputs)
        # Calculate MSE loss
        loss = 100*torch.mean((outputs - targets) ** 2 * weights)
        return loss
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super(PositionalEncoding, self).__init__()

        # Calculate positional encodings
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pos_encoding = torch.zeros((max_len, dim_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.dropout = nn.Dropout(0.1)

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        # Expand the positional encoding to match the batch size
        pos_encoding = self.pos_encoding[:, :token_embedding.size(1)].expand(token_embedding.size(0), -1, -1)
        
        # Residual connection + pos encoding
        return self.dropout(token_embedding + pos_encoding)
    
    
    
class Transformer(nn.Module):
    def __init__(self, d_model=84, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.dim_model = d_model
        self.embedding = nn.Embedding(6, d_model)
        self.positional_encoder = PositionalEncoding(
            dim_model=d_model, max_len=34
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        self.out = nn.Linear(33*84+20+33, 8)

    def forward(self, src, mut_landscape):
        """
        Parameters:
        src: Input matrix of shape (batch_size, seq_len, embedding_dim),
                seq_len=33, embedding_dim=84
        tgt: Target matrix of shape (batch_size, seq_len, embedding_dim)"""
        orig_src = src.clone()
        src = src.to(torch.long)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        output = self.transformer_encoder(src) #(batch_size, 34, 84)
        output = output.view(-1, self.num_flat_features(output)) #(batch_size, 34*84)
        output = torch.cat((output, mut_landscape, orig_src), dim=1) #(batch_size, 34*84+20+33)
        output = self.out(output)
        return output
    
    
    def get_tgt_mask(self, size, device) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones((size, size), device=device) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        #mask = torch.triu(torch.ones((size, size), device=device, dtype=torch.bool), diagonal=1)

        return mask
    
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features