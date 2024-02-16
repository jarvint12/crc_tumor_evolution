import os
import numpy as np
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import lr_scheduler

from sklearn.utils import class_weight
from sklearn.model_selection import KFold


import wandb

from math import log10, floor
import random
from datetime import datetime

from process_input.input_to_3Dmatrix import parse_matrices, FeatureDataset
from model_skeletons.cnn.cnn_v1_16 import cnn, model_version
from compute_accuracy import compute_accuracy
from class_weights_cv import get_class_weights

data_mode = "clu4" #"clu3" "clu431", "clu4" "clu1"
norm=True
project_name = 'cnn_{}'.format(data_mode)
model_script=os.path.join("model_skeletons", "cnn", "{}.py".format(model_version)) #Used to read the file for logs
assert os.path.isfile(model_script), "Could not find file '{}'".format(model_script)
new_files_creation="create_new_data_files.py" #Used to read the file for logs
input_to_matrix=os.path.join("process_input","input_to_3Dmatrix.py") #Used to read the file for logs
assert os.path.isfile(input_to_matrix), "Could not find file '{}'".format(input_to_matrix)
compute_accuracy_file = "compute_accuracy.py"
assert os.path.isfile(compute_accuracy_file), "Could not find file '{}'".format(compute_accuracy_file)
valid_batch_size=5
log_file=os.path.join("logs", data_mode, "{}.log".format(model_version))
cv_log_file=os.path.join("logs", data_mode, "cv_{}.log".format(model_version))
last_run_log_file=os.path.join("logs", data_mode, "current_run_log_{}.log".format(model_version))
last_run_log_file_final_runs=os.path.join("logs", data_mode, "{}_all_runs.log".format(model_version))
model_path=os.path.join("saved_models", data_mode, "{}.pth".format(model_version))
cv=False
train=True


#device = torch.device('cpu')
device = torch.device('cuda')
many_classes=True #ACGT order in onehot

def write_to_log_file(learning_rate, optimizer_type, weight_decay, momentum, is_scheduler, T_max, smoothing, 
                      train_batch_size, valid_batch_size, log_file, model_version, 
                      input_to_matrix, new_greatest_valid_acc, model_path, many_classes,
                      compute_accuracy_file, model_script, run_name, dt_string, norm, acc_list=None):
    """Called, when new highest validation accuracy is found.
    
    Writes everything important information to a log file."""
    with open(log_file, 'w+') as fw:
        fw.write("Created: {}\nModel version: {}\nPath: {}\nRun name: {}\nAccuracy: {}\n\n".format(dt_string, model_version, 
                                                                                                   model_path, run_name, 
                                                                                                   new_greatest_valid_acc))
        if acc_list:
            for k, accuracy in enumerate(acc_list):
                fw.write("Fold {}: {}.\n".format(k+1, accuracy))
            fw.write('\n')
        fw.write("Hyperparameters:\nOptimizer: {}\n".format(optimizer_type))
        if optimizer_type=="SGD":
            fw.write("Momentum: {}\n".format(momentum))
        fw.write("Learning rate: {}\nWeight decay: {}\n".format(learning_rate, weight_decay))
        if is_scheduler:
            fw.write("Used CosineAnnealingLR scheduler with T_max {}\n".format(T_max))
        if many_classes:
            fw.write("Used CrossEntropyLoss with label smoothing {}\n".format(smoothing))
            fw.write("Balanced classes\n")
        else:
            fw.write("Used BCELoss without label smoothing\n")
        fw.write("Data normalized: {}\n".format(norm))
        fw.write("Train batch size: {}\nValidation batch size: {}\n\n".format(train_batch_size, valid_batch_size))
        
        fw.write("\n--------------------------Script of the model can be seen below.---------------------------\n")
        with open(model_script, 'r') as fr:
            fw.write(fr.read())
        fw.write("\n-------------------------------------------------------------------------------------------")
        fw.write("\n\n\n")
    
        fw.write("\n------------------------------Created input matrices with script:------------------------------\n")
        with open(input_to_matrix, 'r') as fr:
            fw.write(fr.read())
        fw.write("\n-------------------------------------------------------------------------------------------")
        fw.write("\n\n\n")
        
        fw.write("\n------------------------------Computed accuracies with script:-----------------------------\n")
        with open(compute_accuracy_file, 'r') as fr:
            fw.write(fr.read())
        fw.write("\n-------------------------------------------------------------------------------------------\n")
        
        
"""In input file, 33 base, 17th is altering, 1st A, 2nd C, 3rd G, 4th T"""
class_types=['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG',
            'GT', 'TA', 'TC', 'TG', 'TT']
classes={'CA': 0, 'CC': 1, 'CG': 2, 'CT': 3, 'TA': 4, 'TC': 5, 'TG': 6, 'TT': 7}

test_file="data/clu9/sompred_crc9_clu9_pyri_mut_combined_test.matrix" #Same with every mode
train_file="data/{0}/sompred_crc9_{0}_pyri_mut_combined_train.matrix".format(data_mode)
valid_file="data/{0}/sompred_crc9_{0}_pyri_mut_combined_valid.matrix".format(data_mode)

trainset_input, trainset_target = parse_matrices(train_file, classes, norm)
valid_input, valid_target = parse_matrices(valid_file, classes, norm)
test_input, test_target = parse_matrices(test_file, classes, norm)

train_dataset=FeatureDataset(data=trainset_input, labels=trainset_target)
valid_dataset=FeatureDataset(data=valid_input, labels=valid_target)
test_dataset=FeatureDataset(data=test_input, labels=test_target)
combined_valid_train = ConcatDataset([train_dataset, valid_dataset]) #Combines validation and training datasets for cross validation

def cv_train_network(net,criterion, valid_criterion,epochs,optimizer,trainloader, validloader,
                  is_scheduler,scheduler, wandb, early_stop=100):
    j=0
    greatest_acc=0
    min_tot_loss=float('inf')
    tot_loss=0
    tot_items=0
    for i in range(epochs):
        net.train()
        for sequences, labels in trainloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            out=net(sequences)
            out=out.squeeze()
            loss=criterion(out,labels)
            tot_loss+=loss.item()
            tot_items+=len(labels)
            loss.backward()
            if torch.isnan(loss):
                raise RuntimeError("NAN!")
            optimizer.step()
        if is_scheduler:
            scheduler.step()
        tot_loss/=tot_items
        accuracy, tot_valid_loss = compute_accuracy(device, net, validloader, valid_criterion, 
                                                "VALID", verbose = False, cv=True)
        if wandb!=None:
            wandb.log({"Training loss": tot_loss,
                       "Validation loss": tot_valid_loss,
                       "Valid Accuracy": accuracy,
            #           "Test loss": test_loss,
            #           "Test Accuracy": test_accuracy,
            #           "Pooled test recall": fake_recall_test,
            #           "Pooled test precision": fake_precision_test,
            #           "Learning rate": optimizer.param_groups[0]['lr'],
            #           "Scheduler": is_scheduler,
                       "Epoch": i})
        if round(accuracy,3)<=round(greatest_acc,3):
            pass
        else:
            greatest_acc=accuracy
        if round(tot_valid_loss,3)>=round(min_tot_loss,3):
            j+=1
            if j>=early_stop and i>100:
                break
        else:
            j=0
            min_tot_loss=tot_valid_loss
        
    return greatest_acc

def train_network(net,criterion, valid_criterion,epochs,optimizer,trainloader, validloader,
                  is_scheduler,scheduler, greatest_acc_overall, model_path, wandb, early_stop=50):
    j=0
    greatest_acc=0
    tot_loss=0
    tot_items=0
    for i in range(epochs):
        net.train()
        for sequences, labels in trainloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            out=net(sequences)
            out=out.squeeze()
            loss=criterion(out,labels)
            tot_loss+=loss.item()
            tot_items+=len(labels)
            loss.backward()
            if torch.isnan(loss):
                raise RuntimeError("NAN!")
            optimizer.step()
        if is_scheduler:
            scheduler.step()
        tot_loss/=tot_items
        accuracy, tot_valid_loss = compute_accuracy(device, net, validloader, valid_criterion, "VALID", verbose = False, cv = True)
        
        if wandb!=None:
            wandb.log({"Training loss": tot_loss,
                       "Validation loss": tot_valid_loss,
                       "Valid Accuracy": accuracy,
            #           "Test loss": test_loss,
            #           "Test Accuracy": test_accuracy,
            #           "Pooled test recall": fake_recall_test,
            #           "Pooled test precision": fake_precision_test,
            #           "Learning rate": optimizer.param_groups[0]['lr'],
            #           "Scheduler": is_scheduler,
                       "Epoch": i})
            
        if round(accuracy,3)<=round(greatest_acc,3):
            if early_stop:
                j+=1
                if j>=early_stop and i>100:
                    print("Greates validation acc: {}".format(greatest_acc))
                    break
        else:
            if accuracy>greatest_acc_overall:
                torch.save(net.state_dict(), model_path)
                greatest_acc_overall=accuracy
            j=0
            greatest_acc=accuracy
    print("Greatest accuracy on run: {}".format(greatest_acc))
    return greatest_acc

def get_earlier_accuracy(log_file):
    with open(log_file, 'r') as fr:
        for line in fr:
            if "Accuracy:" in line:
                return float(line.strip().split(' ')[1]) #Accuracy is written as Accuracy: <acc>
            
greatest_avg_valid_acc = 0
if os.path.isfile(cv_log_file):
    greatest_avg_valid_acc = get_earlier_accuracy(cv_log_file)
print(greatest_avg_valid_acc)

if not os.path.isfile(last_run_log_file):
    with open(last_run_log_file, 'w+') as fw:
        fw.write("Run log.\n\n")
        
#k_fold cross_validation for hyperparameters
if cv:
    k_folds = 5
    epochs=5000
    run_name="None"
    testloader=None

    kfold = KFold(n_splits=k_folds, shuffle=True) #batch size affects the size of datasets
    for i in range(150): #Test with 30 different hyperparameter combinations
        valid_accuracies = list()
        mom_text=""
        scheduler_text=""
        momentum=None
        T_max=None
        is_scheduler=False
        scheduler=None

        learning_rate=random.sample([0.01, 0.001, 0.0001, 0.00001], 1)[0]
        lr_text=str(learning_rate).replace(".","d")
        train_batch_size=random.sample([32, 64, 128], 1)[0]

        optimizer_type=random.sample(["Adam","SGD"], 1)[0]
        if optimizer_type=="SGD":
            momentum= random.sample([0, np.random.uniform()], 1)[0]
            if momentum!=0: momentum = round(momentum, -int(floor(log10(momentum))) + 2)
            mom_text="_mom"+str(round(momentum,2)).replace(".","d")
            is_scheduler=random.sample([True, False], 1)[0]
            if is_scheduler:
                T_max=random.sample([1, np.random.uniform(low=0.2)], 1)[0]
                T_max = round(T_max, -int(floor(log10(T_max))) + 2)
        weight_decay=random.sample([0, 0.00001, 0.0001, 0.001], 1)[0]
        if weight_decay!=0: weight_decay = round(weight_decay, -int(floor(log10(weight_decay))) + 2)
        decay_text="_wdecay"+str(weight_decay).replace(".","d")
        smoothing=random.sample([0, np.random.uniform(high=0.05)], 1)[0]
        if smoothing!=0: smoothing = round(smoothing, -int(floor(log10(smoothing))) + 2)

        for (train_ids, test_ids) in kfold.split(combined_valid_train):
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(
                          combined_valid_train, 
                          batch_size=train_batch_size, sampler=train_subsampler)
            validloader = torch.utils.data.DataLoader(
                              combined_valid_train,
                              batch_size=valid_batch_size, sampler=test_subsampler)
            net = cnn().to(device)
            if optimizer_type=="Adam":
                optimizer=torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type=="SGD":
                optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, 
                                            momentum=momentum, weight_decay=weight_decay)
                if is_scheduler:
                    scheduler=lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max*epochs)
            else:
                raise RuntimeError("WRONG OPTIMIZER: {}".format(optimizer_type))


            train_class_weights, valid_class_weights = get_class_weights(device, trainloader, validloader)
            criterion=nn.CrossEntropyLoss(weight=train_class_weights, label_smoothing=smoothing)
            valid_criterion=nn.CrossEntropyLoss(weight=valid_class_weights)

            valid_acc = cv_train_network(net,criterion,valid_criterion,epochs,optimizer,trainloader, validloader, 
                                      is_scheduler,scheduler, wandb=None)
            valid_accuracies.append(valid_acc)
        avg_valid_acc = sum(valid_accuracies) / len(valid_accuracies)

        print("Optimizer: {}\nLearning Rate: {}\nScheduler: {}".format(optimizer_type, learning_rate, is_scheduler))
        print("Weight decay: {}\nSmoothing: {}".format(weight_decay, smoothing))
        if optimizer_type=="SGD":
            print("Momentum: {}".format(momentum))
            if is_scheduler:
                print("T_max: {}\n".format(T_max))
        print("Average validation accuracy: {}".format(avg_valid_acc))
        for k, accuracy in enumerate(valid_accuracies):
            print("Fold {}: {}.".format(k+1, accuracy))

        with open(last_run_log_file, 'a+') as fw:
            fw.write("Time: {}\n".format(datetime.now().strftime("%d.%m.%Y %H:%M:%S")))
            fw.write("Average validation accuracy: {}\n".format(avg_valid_acc))
            fw.write("\n".join(["Fold {}: {}.".format(k+1, accuracy) for k, accuracy in enumerate(valid_accuracies)]))
            fw.write("\nOptimizer: {}\nLearning Rate: {}\nScheduler: {}\n".format(optimizer_type, learning_rate, is_scheduler))
            fw.write("Weight decay: {}\nSmoothing: {}\nBatch size: {}\n".format(weight_decay, smoothing, train_batch_size))
            if optimizer_type=="SGD":
                fw.write("Momentum: {}\n".format(momentum))
                if is_scheduler:
                    fw.write("T_max: {}\n".format(T_max))
            fw.write('\n\n')

        if avg_valid_acc>greatest_avg_valid_acc:
            now = datetime.now()
            dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
            write_to_log_file(learning_rate, optimizer_type, weight_decay, momentum, is_scheduler, T_max,
                             smoothing, train_batch_size, valid_batch_size, cv_log_file, model_version, 
                              input_to_matrix, 
                              avg_valid_acc, model_path, many_classes, compute_accuracy_file,
                              model_script, run_name, dt_string, acc_list=valid_accuracies, norm=norm)
            greatest_avg_valid_acc=avg_valid_acc


greatest_acc_overall=0
if os.path.isfile(log_file):
    greatest_acc_overall=get_earlier_accuracy(log_file)
print(greatest_acc_overall)
train_batch_size=64

trainloader = torch.utils.data.DataLoader(train_dataset
    ,batch_size=train_batch_size
    ,shuffle=True
    ,drop_last=True
)
validloader = torch.utils.data.DataLoader(valid_dataset
    ,batch_size=5
    ,shuffle=False
)

if many_classes:
    class_weights=class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(trainset_target),
        y=np.array(trainset_target))
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    class_weights=class_weights.to(device)
    
    valid_class_weights=class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(valid_target),
        y=np.array(valid_target))
    valid_class_weights=torch.tensor(valid_class_weights,dtype=torch.float)
    valid_class_weights=valid_class_weights.to(device)
    
else:
    print(np.array(trainset_target))
print(class_weights, valid_class_weights)

if train:
    learning_rate=0.001
    momentum=0.193
    T_max=0.715
    is_scheduler=True
    optimizer_type="SGD"
    weight_decay=0
    smoothing=0.0466
    scheduler=None
    epochs=5000
    criterion=nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smoothing)
    valid_criterion=nn.CrossEntropyLoss(weight=valid_class_weights)
    lr_text=str(learning_rate).replace(".","d")
    decay_text="_wdecay"+str(weight_decay).replace(".","d")
    mom_text=""
    scheduler_text=""
    for i in range(1000):
        net = cnn().to(device)
        if optimizer_type=="Adam":
            optimizer=torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type=="SGD":
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, 
                                        momentum=momentum, weight_decay=weight_decay)
            mom_text="_mom"+str(round(momentum,2)).replace(".","d")
        else:
            raise RuntimeError("WRONG OPTIMIZER.")
        if is_scheduler:
            scheduler=lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max*epochs)
            scheduler_text="-cos"+str(T_max).replace(".","d")
        
        run = wandb.init(project=project_name)
        run_name="run_{}".format(i)
        wandb.run.name = run_name
        config = wandb.config
        config.batch_size=train_batch_size

        greatest_acc = train_network(net,criterion, valid_criterion,epochs,optimizer,trainloader, validloader,
                          is_scheduler,scheduler, greatest_acc_overall, model_path, wandb, early_stop=None)

        with open(last_run_log_file_final_runs, 'a+') as fw:
                fw.write("Time: {}\n".format(datetime.now().strftime("%d.%m.%Y %H:%M:%S")))
                fw.write("Accuracy: {}\n".format(greatest_acc))
                fw.write("\nOptimizer: {}\nLearning Rate: {}\nScheduler: {}\n".format(optimizer_type, learning_rate, is_scheduler))
                fw.write("Weight decay: {}\nSmoothing: {}\nBatch size: {}\n".format(weight_decay, smoothing, train_batch_size))
                if optimizer_type=="SGD":
                    fw.write("Momentum: {}\n".format(momentum))
                    if is_scheduler:
                        fw.write("T_max: {}\n".format(T_max))
                fw.write('\n\n')


        if greatest_acc>greatest_acc_overall:
            now = datetime.now()
            dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
            write_to_log_file(learning_rate, optimizer_type, weight_decay, momentum, is_scheduler, T_max,
                             smoothing, train_batch_size, valid_batch_size, log_file, model_version, 
                              input_to_matrix, 
                              greatest_acc, model_path, many_classes, compute_accuracy_file,
                              model_script, run_name="None", dt_string=dt_string, norm=norm)
            greatest_acc_overall=greatest_acc
                #raise Exception("LOL")
                        #run.finish()
    #learning_rate=0.001
    #momentum=0.9
    #gamma=2
    #is_scheduler=True
    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)#, weight_decay=0.1-0.0001
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)
            
testloader = DataLoader(test_dataset
    ,batch_size=5
    ,shuffle=False
)

model=cnn()
#model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path))
print(model_path)
model.eval()
model.to(device)

criterion=nn.CrossEntropyLoss(weight=class_weights)
accuracy, loss = \
compute_accuracy(device, model, testloader, None, "TEST", verbose=True, cv=False)

print("Accuracy:",accuracy)

accuracy, tot_valid_loss = compute_accuracy(device, model, validloader, None, "VALID", verbose = True, cv = False)

print("Accuracy:",accuracy)