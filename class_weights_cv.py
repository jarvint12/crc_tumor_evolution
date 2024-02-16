import torch
import numpy as np
from sklearn.utils import class_weight


def get_class_weights(device, trainloader, validloader):
    trainset_target=torch.cat([labels for _, labels in trainloader], dim=0)
    train_class_weights=class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(trainset_target.cpu()),
                        y=np.array(trainset_target.cpu()))
    train_class_weights=torch.tensor(train_class_weights,dtype=torch.float)
    train_class_weights=train_class_weights.to(device)
    
    valid_target=torch.cat([labels for _, labels in validloader], dim=0)
    valid_class_weights=class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(valid_target.cpu()),
                        y=np.array(valid_target.cpu()))
    valid_class_weights=torch.tensor(valid_class_weights,dtype=torch.float)
    valid_class_weights=valid_class_weights.to(device)
    return train_class_weights, valid_class_weights




def get_class_weights_mut_landscape(device, trainloader, validloader):
    #for _, _, _, labels in trainloader:
    #    print(labels)
    trainset_target=torch.cat([labels for _, _, _, labels in trainloader], dim=0)
    train_class_weights=class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(trainset_target.cpu()),
                        y=np.array(trainset_target.cpu()))
    train_class_weights=torch.tensor(train_class_weights,dtype=torch.float)
    train_class_weights=train_class_weights.to(device)
    
    valid_target=torch.cat([labels for _, _, _, labels in validloader], dim=0)

    valid_class_weights=class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(valid_target.cpu()),
                        y=np.array(valid_target.cpu()))
    valid_class_weights=torch.tensor(valid_class_weights,dtype=torch.float)
    valid_class_weights=valid_class_weights.to(device)
    return train_class_weights, valid_class_weights

def main():
    raise RuntimeError("No main implemented!")

if __name__=='__main__':
    main()