import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    def __init__(self, data, labels):
        '''
        Args:
        info_dir (string/pandas Dataframe): Path to excel file(or the file itself), that contains clinical info about PET images
        seed (int): seed for sampling images
        norm_mean_std (str): string that indicates the means and stds for normalization
        prob_gaussian (float): Probability for blurring images
        simple_transformation (bool): Whether to use minimal transformations
        
        Outputs:
        image (torch.Tensor): Image as torch Tensor. Shape (1,3,512,512)
        label (torch.Tensor): Label indicating if there is cancer in the picture. 1=Cancer, 0=Benign 
        '''
        self.data_tensor=data
        self.labels=labels

    def __len__(self):
        return self.data_tensor.shape[0] 
    
    def __getitem__(self, idx):
        #load images
        tensor = self.data_tensor[idx,:,:]
        #load labels 
        label = self.labels[idx]
        
        return tensor, label
    
    
def get_norm_values(file):
    numbers=dict()
    previous_id=None
    with open(file, 'r') as fr:
        for line in fr:
            if line.startswith('#'):
                if line.startswith("#ID:"):
                        if line.strip()!=previous_id:
                            previous_id=line.strip()
                            channel=0
                        else:
                            channel+=1
                continue
            if not channel in numbers:
                numbers[channel]=list()
            numbers[channel] += [float(number) for number in line.strip().split(',')]
    mins=dict()
    maxes=dict()
    for channel in numbers:
        mins[channel]=min(numbers[channel])
        maxes[channel]=max(numbers[channel])
    return mins, maxes


def parse_matrices(file, classes, norm):
    if norm:
        mins, maxes = get_norm_values(file)

    first_row_of_sample=True
    first_sample=True
    df_input=None
    new_sampleid=True
    previous_id=None
    target=list()
    with open(file, 'r') as fr:
        for line in fr:
            if line.startswith('#'):
                if line.startswith("#ID:"):
                    if previous_id==None:
                        previous_id=line.strip()
                        channel=0
                        continue
                    elif line.strip()!=previous_id:
                        new_sampleid=True
                        channel=0
                        if first_sample:
                            df_input=temp_2d
                            first_sample=False
                        else:
                            df_input=np.concatenate((df_input, temp_2d), axis=0)
                        first_row_of_sample=True
                        previous_id=line.strip()
                    else:
                        channel+=1
                if new_sampleid:
                    if line.startswith('#REF:'):
                        ref=line[6]
                    elif line.startswith('#ALT:'):
                        alt=line[6]
                        target.append(torch.tensor(classes[ref+alt]))
                        new_sampleid=False
                continue
            if norm:
                numbers = [(float(number)-mins[channel])/(maxes[channel]-mins[channel]) for number in line.strip().split(',')]
            else:
                numbers= [float(number) for number in line.strip().split(',')]
            row_in_array=np.array(numbers).reshape(1,1,-1,1)
            if first_row_of_sample:
                temp_2d=row_in_array
                first_row_of_sample=False
            else:
                temp_2d=np.concatenate((temp_2d, row_in_array), axis=3)

    if not first_row_of_sample:
        df_input=np.concatenate((df_input, temp_2d), axis=0)
    return torch.Tensor(df_input), torch.tensor(target,dtype=torch.long)
    
    
if __name__=='__main__':
    raise RuntimeError()