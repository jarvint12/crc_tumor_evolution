import torch

def normalize(trainset_input, valid_input, test_input):
    """Normalizes by each channel's (4x33) maximum value"""
    for i in range(trainset_input.shape[1]):
        if i==0:
            trainset_input2=(trainset_input[:,i,:,:]/trainset_input[:,i,:,:].max()).reshape(trainset_input.shape[0],
                                                                                     1, trainset_input.shape[2],
                                                                                         trainset_input.shape[3])
            print("Training set original shape: {}.\nTraining set normalized shape for one channel: {}.".format(trainset_input.shape,trainset_input2.shape))
        else:
            temp=(trainset_input[:,i,:,:]/trainset_input[:,i,:,:].max()).reshape(trainset_input.shape[0],
                                                                                     1, trainset_input.shape[2],
                                                                                         trainset_input.shape[3])
            trainset_input2=torch.cat((trainset_input2,temp), dim=1)

    for i in range(valid_input.shape[1]):
        if i==0:
            valid_input2=(valid_input[:,i,:,:]/valid_input[:,i,:,:].max()).reshape(valid_input.shape[0],
                                                                                     1, valid_input.shape[2],
                                                                                         valid_input.shape[3])
            print("Validation set original shape: {}.\nTraining set normalized shape for one channel: {}.".format(valid_input.shape,valid_input2.shape))
        else:
            temp=(valid_input[:,i,:,:]/valid_input[:,i,:,:].max()).reshape(valid_input.shape[0],
                                                                                     1, valid_input.shape[2],
                                                                                         valid_input.shape[3])
            valid_input2=torch.cat((valid_input2,temp), dim=1)


    for i in range(test_input.shape[1]):
        if i==0:
            test_input2=(test_input[:,i,:,:]/test_input[:,i,:,:].max()).reshape(test_input.shape[0],
                                                                                     1, test_input.shape[2],
                                                                                         test_input.shape[3])
            print("Test set original shape: {}.\nTraining set normalized shape for one channel: {}.".format(test_input.shape,test_input2.shape))
        else:
            temp=(test_input[:,i,:,:]/test_input[:,i,:,:].max()).reshape(test_input.shape[0],
                                                                                     1, test_input.shape[2],
                                                                                         test_input.shape[3])
            test_input2=torch.cat((test_input2,temp), dim=1)
    return trainset_input2, valid_input2, test_input2