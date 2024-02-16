from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        tensor = self.data_tensor[idx,:,:,:]
        #load labels 
        label = self.labels[idx]
        
        return tensor, label
   

def create_batches(train_dataset, valid_dataset, test_dataset, train_batch_size, valid_batch_size):
    #train_dataset=ConcatDataset([train_dataset, valid_dataset])  #Poista
    trainloader = DataLoader(train_dataset
        ,batch_size=train_batch_size#128#32
        ,shuffle=True
        ,drop_last=True
    )
    
    validloader =  DataLoader(valid_dataset
        ,batch_size=valid_batch_size#128
        ,shuffle=False
    )
    
    
    
    testloader = DataLoader(test_dataset
        ,batch_size=valid_batch_size
        ,shuffle=False
    )
    
    return trainloader, validloader, testloader
 
def main(trainset_input, valid_input, test_input, trainset_target, valid_target, test_target, train_batch_size, valid_batch_size):
    train_dataset=FeatureDataset(data=trainset_input, labels=trainset_target)
    valid_dataset=FeatureDataset(data=valid_input, labels=valid_target)
    test_dataset=FeatureDataset(data=test_input, labels=test_target)


    trainloader, validloader, testloader = create_batches(train_dataset, valid_dataset, test_dataset, train_batch_size, valid_batch_size)
    
    return trainloader, validloader, testloader
    
    
if __name__=='__main__':
    main()