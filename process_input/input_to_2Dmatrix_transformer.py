import numpy as np
import torch
from torch.utils.data import Dataset
import collections

class FeatureDataset(Dataset):
    def __init__(self, data, targets, labels):
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
        self.targets=targets

    def __len__(self):
        return len(self.data_tensor)
    
    def __getitem__(self, idx):
        #load images
        tensor = self.data_tensor[idx]
        #load labels 
        label = self.labels[idx]
        #load targets
        target = self.targets[idx]
        return tensor, label, target


def create_class_weights(class_amounts, total_amount, norm_type):
    class_weights=collections.Counter()
    for target_class in class_amounts:
        weight = total_amount / (len(class_amounts) * class_amounts[target_class]) #tot_samples/(n_classes*n_samples_in_class)
        class_weights[target_class] = weight
    if norm_type=="sum":
        # normalize by sum
        sum_weights = sum(class_weights.values())
        for target_class in class_weights:
            class_weights[target_class] = class_weights[target_class] / sum_weights
    elif norm_type=="max":
        # normalize by max
        max_weight = max(class_weights.values())
        for target_class in class_weights:
            class_weights[target_class] = class_weights[target_class] / max_weight
    return class_weights

    
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


def create_target(target_mode, temp_2d, target_row, first_genome_row):
    temp_target = np.copy(temp_2d) #(1, 33, 84)
    for i in range(first_genome_row, first_genome_row+4):
        temp_target[0, 16, i]=int(i==first_genome_row+target_row) #Change target base to 1 and every other to 0
    if target_mode not in ["whole_matrix", "target_with_landscape"]:
        
        for i in range(temp_target.shape[2]): #Go through every row in matrix
            if first_genome_row<=i<first_genome_row+4: #Skip DNA sequence rows
                continue
            for j in range(33): #Go through every col in row
                temp_target[0, j, i] = 0 #Change all values to 0, so only DNA sequences are chosen
    if target_mode in ["only_target_base", "target_with_landscape"]:
        temp_target=temp_target[0,16,:].reshape(1,1,-1)
    return temp_target

def parse_matrices(file, norm, target_mode, norm_type):
    assert target_mode in ["whole_matrix", "whole_DNA_seq", "only_target_base", "target_with_landscape"], \
                            "Incorrect target mode {}".format(target_mode)
    first_genome_row=80
    if norm:
        mins, maxes = get_norm_values(file)
    bases={'A': 0, 'C': 1, 'G': 2, 'T': 3}
    classes={'CA': 0, 'CC': 1, 'CG': 2, 'CT': 3, 'TA': 4, 'TC': 5, 'TG': 6, 'TT': 7} #For class weights
    class_amounts=collections.Counter() #For class weights
    total_amount=0 #For class weights
    first_row_of_sample=True
    first_sample=True
    df_input=None
    labels=list()
    new_sampleid=True
    previous_id=None
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
                        temp_target = create_target(target_mode, temp_2d, target_row, first_genome_row)
                        
                        if first_sample:
                            df_input=temp_2d
                            df_target=temp_target
                            first_sample=False
                        else:
                            df_input=np.concatenate((df_input, temp_2d), axis=0)
                            df_target=np.concatenate((df_target, temp_target), axis=0) #(batch_size, 33, 84)
                        first_row_of_sample=True
                        previous_id=line.strip()
                    else:
                        channel+=1
                if new_sampleid and line.startswith('#ALT:'):
                        alt=line[6]
                        target_row = bases[alt]
                        new_sampleid=False
                        class_amounts[classes[ref+alt]]+=1
                        labels.append(torch.tensor(classes[ref+alt]))
                        total_amount+=1
                elif line.startswith('#REF:'):
                        ref=line[6]
                continue
            if norm:
                numbers = [(float(number)-mins[channel])/(maxes[channel]-mins[channel]) for number in line.strip().split(',')]
            else:
                numbers= [float(number) for number in line.strip().split(',')]
            row_in_array=np.array(numbers).reshape(1,-1,1)
            if first_row_of_sample:
                temp_2d=row_in_array
                first_row_of_sample=False
            else:
                temp_2d=np.concatenate((temp_2d, row_in_array), axis=2)

    if not first_row_of_sample:
        df_input=np.concatenate((df_input, temp_2d), axis=0)
        temp_target = create_target(target_mode, temp_2d, target_row, first_genome_row)
        df_target=np.concatenate((df_target, temp_target), axis=0)
        
    class_weights = create_class_weights(class_amounts, total_amount, norm_type=norm_type)
    return torch.Tensor(df_input), torch.Tensor(df_target), class_weights, torch.tensor(labels,dtype=torch.long)
    
    
if __name__=='__main__':
    raise RuntimeError()