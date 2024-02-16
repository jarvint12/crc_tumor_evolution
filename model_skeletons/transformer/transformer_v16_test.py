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
    

def change_to_onehot(old_tensor):
    tensor = torch.zeros((old_tensor.shape[0], 34, 84))

    # Replace the values in the tensor based on the given conditions
    tensor[:, :, 0] = (old_tensor == 0).float()
    tensor[:, :, 80] = (old_tensor == 1).float()
    tensor[:, :, 81] = (old_tensor == 2).float()
    tensor[:, :, 82] = (old_tensor == 3).float()
    tensor[:, :, 83] = (old_tensor == 4).float()

    # Remove the last dimension from the tensor
    #tensor = tensor[:, :, :-1]
    tensor = tensor.to(old_tensor.get_device())
    return tensor

class Transformer(nn.Module):
    def __init__(self, d_model=84, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Transformer, self).__init__()
        self.dim_model = d_model
        self.embedding = nn.Embedding(6, d_model)
        self.positional_encoder = PositionalEncoding(
            dim_model=self.dim_model, max_len=34
        )
        self.transformer = nn.Transformer(d_model=self.dim_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                         batch_first=True)
        #self.out = nn.Linear(34*84+20+33, 8)
        self.out = nn.Linear(2856, 8)

    def forward(self, orig_src, tgt, mut_landscape, tgt_mask=None, evaluate=False):
        """
        Parameters:
        src: Input matrix of shape (batch_size, seq_len+1),
                seq_len=33, embedding_dim=84
        tgt: Target matrix of shape (batch_size, seq_len+1)"""
        src = orig_src.to(torch.long)
        src = change_to_onehot(src)
        tgt = tgt.to(torch.long)
        tgt = change_to_onehot(tgt)
#         if evaluate:
#             tgt[:,17]=src[:,16]
#             print("src",orig_src)
#             print("mask", tgt_mask)
#             print("Landscape",mut_landscape)
#             print("Tgt",tgt)
        #src = self.embedding(src) * math.sqrt(self.dim_model)
        #tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt) #
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = output.view(-1, self.num_flat_features(output)) #(batch_size, 34*84)
        #output = torch.cat((output, mut_landscape), dim=1) #(batch_size, 34*84+20)
        #output = torch.cat((output, mut_landscape, orig_src[:,:-1]), dim=1) #(batch_size, 34*84+20+33), leave END token out
        output = self.out(output)
        return output
    
    
    def get_tgt_mask(self, size, device) -> torch.tensor:
    # Generates a square matrix where each row allows one word more to be seen
        tgt_mask = torch.triu(torch.ones((size, size), device=device) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        return tgt_mask
    
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features