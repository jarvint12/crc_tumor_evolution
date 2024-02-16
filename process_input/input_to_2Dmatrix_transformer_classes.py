import numpy as np
import torch
from torch.utils.data import Dataset
import copy

class FeatureDataset(Dataset):
    def __init__(self, data, targets, mut_landscape, labels):
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
        self.targets=targets
        self.mut_landscape = mut_landscape
        self.labels=labels

    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        #load sequences
        tensor = self.data_tensor[idx]
        #load targets
        targets = self.targets[idx]
        #load mutational landscape
        mut_landscape = self.mut_landscape[idx]
        #load labels 
        label = self.labels[idx]
        return tensor, targets, mut_landscape, label

    
def get_norm_values(file):
    numbers=dict()
    previous_id=None
    with open(file, 'r') as fr:
        for line in fr:
            if line.startswith('#'):
                if line.startswith("#ID:"):
                        if line.strip()!=previous_id:
                            previous_id=line.strip()
                            channel_number=0
                        else:
                            channel_number+=1
                continue
            if not channel_number in numbers:
                numbers[channel_number]=list()
            numbers[channel_number] += [float(number) for number in line.strip().split(',')]
    mins=dict()
    maxes=dict()
    for channel_number in numbers:
        mins[channel_number]=min(numbers[channel_number])
        maxes[channel_number]=max(numbers[channel_number])
    return mins, maxes


def add_sample_to_data(sequence, target_base, mutational_landscape, df_input, df_target, df_mut_landscape):
    target_sequence = copy.copy(sequence)
    target_sequence[16] = target_base
    sequence = np.array(sequence).reshape(1,-1)
    target_sequence = np.array(target_sequence).reshape(1,-1)
    mutational_landscape = np.array(mutational_landscape).reshape(1,-1)

    if not isinstance(df_input, np.ndarray):
        df_input=sequence
        df_target=target_sequence
        df_mut_landscape = mutational_landscape
    else:
        df_input=np.concatenate((df_input, sequence), axis=0)
        df_target=np.concatenate((df_target, target_sequence), axis=0) #(batch_size, 33)
        df_mut_landscape=np.concatenate((df_mut_landscape, mutational_landscape), axis=0)
    return df_input, df_target, df_mut_landscape



def parse_matrices(file, norm, onehot=False):
    if norm:
        mins, maxes = get_norm_values(file)
    bases={'A': 0, 'C': 1, 'G': 2, 'T': 3}
    classes={'CA': 0, 'CC': 1, 'CG': 2, 'CT': 3, 'TA': 4, 'TC': 5, 'TG': 6, 'TT': 7} #For class weights
    first_sample=True
    df_input=None
    labels=list()
    previous_id=None
    with open(file, 'r') as fr:
        for line in fr:
            if line.startswith('#'):
                if line.startswith("#ID:"):
                    if previous_id==None:
                        target_collected=False
                        previous_id=line.strip()
                        channel_number=-1
                        sequence = [0 for _ in range(33)]
                        mutational_landscape = [0 for _ in range(20)]
                        previous_id=line.strip()
                        continue
                    elif line.strip()!=previous_id:
                        if first_sample:
                            df_input, df_target, df_mut_landscape = add_sample_to_data(sequence, target_base, mutational_landscape, None, None, None)
                            first_sample=False
                        else:
                            df_input, df_target, df_mut_landscape = add_sample_to_data(sequence, target_base, mutational_landscape, df_input, df_target, df_mut_landscape)
                        target_collected=False
                        channel_number=-1 #Goes to 0 at the first channel
                        sequence = [0 for _ in range(33)]
                        mutational_landscape = [0 for _ in range(20)]
                        previous_id=line.strip()
                        
                if line.startswith('#ALT:') and not target_collected:
                        alt=line[6]
                        target_base = bases[alt]+1 #Start from 1
                        labels.append(torch.tensor(classes[ref+alt]))
                        target_collected=True
                elif line.startswith('#REF:'):
                        ref=line[6]
                elif line.startswith('#CHANNEL:'):
                        current_channel=line.strip().split()[1]#CHANNEL: genome
                        channel_row=0
                        channel_number+=1
                continue

            if norm:
                numbers = [(float(number)-mins[channel_number])/(maxes[channel_number]-mins[channel_number]) for number in line.strip().split(',')]
            else:
                numbers= [float(number) for number in line.strip().split(',')]
            if current_channel.lower() == 'genome':
                for index, number in enumerate(numbers):
                    if number==1:
                        sequence[index]=channel_row+1
            else:
                max_number = max(numbers)
                if mutational_landscape[channel_number]==0 and max_number!=0:
                    mutational_landscape[channel_number]=max_number
            channel_row+=1
                
    
    #Add also the last sample
    df_input, df_target, df_mut_landscape = add_sample_to_data(sequence, target_base, mutational_landscape, df_input, df_target, df_mut_landscape)
    return torch.Tensor(df_input), torch.Tensor(df_target), torch.Tensor(df_mut_landscape), torch.tensor(labels,dtype=torch.long)
    
    
if __name__=='__main__':
    raise RuntimeError()