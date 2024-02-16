import os
import numpy as np
import matplotlib.pyplot as plt

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.optim import lr_scheduler

from sklearn.utils import class_weight
from sklearn.model_selection import KFold

from tqdm import tqdm

import wandb

import collections
from math import log10, floor
import random
from datetime import datetime

from process_input.input_to_2Dmatrix_transformer_masters_thesis import parse_matrices, FeatureDataset, create_class_weights
from model_skeletons.transformer.transformer_v10 import Transformer, model_version, WeightedMSELoss
from compute_accuracy_transformer_masters_thesis import compute_accuracy

new_mode=False #If uses new matrix parsing or old, during masters thesis
if new_mode:
    from process_input.input_to_2Dmatrix_transformer import parse_matrices, FeatureDataset, create_class_weights
    from compute_accuracy_transformer import compute_accuracy
else:
    from process_input.input_to_2Dmatrix_transformer_masters_thesis import parse_matrices, FeatureDataset, create_class_weights
    from compute_accuracy_transformer_masters_thesis import compute_accuracy
data_mode = "clu4" #clu3, #clu4, clu431 clu1"
norm=True
norm_type = None
target_mode="whole_matrix" #["whole_matrix", "whole_DNA_seq", "only_target_base", "target_with_landscape"]
if target_mode in ["whole_matrix", "whole_DNA_seq"]:
    correct_label_index=16
else:
    correct_label_index=0

project_name = 'transformer_{}'.format(data_mode)
model_script=os.path.join("model_skeletons", "transformer", "{}.py".format(model_version)) #Used to read the file for logs
assert os.path.isfile(model_script), "Could not find file '{}'".format(model_script)
new_files_creation="create_new_data_files.py" #Used to read the file for logs
input_to_matrix=os.path.join("process_input","input_to_2Dmatrix_transformer.py") #Used to read the file for logs
assert os.path.isfile(input_to_matrix), "Could not find file '{}'".format(input_to_matrix)
compute_accuracy_file = "compute_accuracy_transformer.py"
assert os.path.isfile(compute_accuracy_file), "Could not find file '{}'".format(compute_accuracy_file)
valid_batch_size=5
log_file=os.path.join("logs", data_mode, "{}.log".format(model_version))
cv_log_file=os.path.join("logs", data_mode, "cv_{}.log".format(model_version))
last_run_log_file=os.path.join("logs", data_mode, "{}_cv_all_runs.log".format(model_version))
last_run_log_file_final_runs=os.path.join("logs", data_mode, "{}_all_runs.log".format(model_version))
model_path=os.path.join("saved_models", data_mode, "{}.pth".format(model_version))
cv=False
train=True
indices_of_interest=[80,81,82,83]



#device = torch.device('cpu')
device = torch.device('cuda')
many_classes=True #ACGT order in onehot

def write_to_log_file(learning_rate, optimizer_type, weight_decay, 
                      train_batch_size, valid_batch_size, log_file, model_version, 
                      input_to_matrix, new_greatest_valid_acc, model_path, many_classes,
                      compute_accuracy_file, model_script, run_name, dt_string, target_mode,
                      nhead, num_encoder_layers, num_decoder_layers, norm, norm_type, acc_list=None):
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
        fw.write("Learning rate: {}\nWeight decay: {}\n".format(learning_rate, weight_decay))
        fw.write("head: {}\nnum_encoder_layers: {}\nnum_decoder_layers: {}\n".format(nhead, num_encoder_layers, num_decoder_layers))
        fw.write("Used MSELoss\n")
        fw.write("Balanced classes\n")
        fw.write("Data normalized: {}\n".format(norm))
        fw.write("Norm type: {}\n".format(norm_type))
        fw.write("Target mode: {}\n".format(target_mode))
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

if new_mode:
    trainset_input, trainset_target, class_weights_train_whole, labels_trainset = parse_matrices(train_file, norm, target_mode, norm_type)
    valid_input, valid_target, class_weights_valid_whole, labels_valid = parse_matrices(valid_file, norm, target_mode, norm_type)
    test_input, test_target, class_weights_test, labels_test = parse_matrices(test_file, norm, target_mode, norm_type)

    sos_token = torch.zeros((1, 1, trainset_input.shape[2]))
    eos_token = torch.ones((1, 1, trainset_input.shape[2]))

    trainset_input_with_tokens = torch.cat([trainset_input, eos_token.expand(trainset_input.size(0), 1, -1)], dim=1)
    trainset_target_with_tokens = torch.cat([sos_token.expand(trainset_input.size(0), 1, -1), trainset_target, eos_token.expand(trainset_input.size(0), 1, -1)], dim=1)

    valid_input_with_tokens = torch.cat([valid_input, eos_token.expand(valid_input.size(0), 1, -1)], dim=1)
    valid_target_with_tokens = torch.cat([sos_token.expand(valid_input.size(0), 1, -1), valid_target, eos_token.expand(valid_input.size(0), 1, -1)], dim=1)

    test_input_with_tokens = torch.cat([test_input, eos_token.expand(test_input.size(0), 1, -1)], dim=1)
    test_target_with_tokens = torch.cat([sos_token.expand(test_input.size(0), 1, -1), test_target, eos_token.expand(test_input.size(0), 1, -1)], dim=1)

    train_dataset=FeatureDataset(data=trainset_input_with_tokens, labels=labels_trainset, targets=trainset_target_with_tokens)
    valid_dataset=FeatureDataset(data=valid_input_with_tokens, labels=labels_valid, targets=trainset_target_with_tokens[:valid_input_with_tokens.shape[0],:,:])
    test_dataset=FeatureDataset(data=test_input_with_tokens, labels=labels_test, targets=test_target_with_tokens)
    combined_valid_train = ConcatDataset([train_dataset, valid_dataset]) #Combines validation and training datasets for cross validation

else:
    trainset_input, trainset_target, class_weights_train_whole = parse_matrices(train_file, norm, target_mode, norm_type)
    valid_input, valid_target, class_weights_valid_whole = parse_matrices(valid_file, norm, target_mode, norm_type)
    test_input, test_target, class_weights_test = parse_matrices(test_file, norm, target_mode, norm_type)

    sos_token = torch.zeros((1, 1, trainset_input.shape[2]))
    eos_token = torch.ones((1, 1, trainset_input.shape[2]))

    trainset_input_with_tokens = torch.cat([trainset_input, eos_token.expand(trainset_input.size(0), 1, -1)], dim=1)
    trainset_target_with_tokens = torch.cat([sos_token.expand(trainset_input.size(0), 1, -1), trainset_target, eos_token.expand(trainset_input.size(0), 1, -1)], dim=1)

    valid_input_with_tokens = torch.cat([valid_input, eos_token.expand(valid_input.size(0), 1, -1)], dim=1)
    valid_target_with_tokens = torch.cat([sos_token.expand(valid_input.size(0), 1, -1), valid_target, eos_token.expand(valid_input.size(0), 1, -1)], dim=1)

    test_input_with_tokens = torch.cat([test_input, eos_token.expand(test_input.size(0), 1, -1)], dim=1)
    test_target_with_tokens = torch.cat([sos_token.expand(test_input.size(0), 1, -1), test_target, eos_token.expand(test_input.size(0), 1, -1)], dim=1)

    train_dataset=FeatureDataset(data=trainset_input_with_tokens, labels=trainset_target_with_tokens)
    valid_dataset=FeatureDataset(data=valid_input_with_tokens, labels=valid_target_with_tokens)
    test_dataset=FeatureDataset(data=test_input_with_tokens, labels=test_target_with_tokens)
    combined_valid_train = ConcatDataset([train_dataset, valid_dataset]) #Combines validation and training datasets for cross validation

def cv_train_network(net,criterion, valid_criterion,epochs,optimizer,trainloader, validloader,
                  correct_label_index, wandb, early_stop=100):
    j=0
    greatest_acc=0
    min_tot_loss=float('inf')
    tot_loss=0
    tot_items=0
    for i in range(epochs):
        net.train()
        for sequences, targets in trainloader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            targets_input = targets[:,:-1]
            targets_expected = targets[:,1:]
            
            sequence_length = targets_input.size(1)
            tgt_mask = net.get_tgt_mask(sequence_length, device)
            
            optimizer.zero_grad()
            out = net(sequences, targets_input, tgt_mask =tgt_mask)
            #out = out.squeeze()
            loss=criterion(sequences, targets_expected, out)
            tot_loss+=loss.item()
            tot_items+=len(labels)
            loss.backward()
            if torch.isnan(loss):
                raise RuntimeError("NAN!")
            optimizer.step()
        tot_loss/=tot_items
        accuracy, tot_valid_loss = compute_accuracy(device, net, validloader, valid_criterion, 
                                                "VALID", verbose = False, cv=True,
                                                   correct_label_index=correct_label_index) #17th/33+Start token
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
                  correct_label_index, greatest_acc_overall, model_path, wandb, early_stop=50):
    j=0
    greatest_acc=0
    tot_loss=0
    tot_items=0
    for i in range(epochs):
        net.train()
        for sequences, targets in trainloader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            targets_input = targets[:,:-1]
            targets_expected = targets[:,1:]
            
            sequence_length = targets_input.size(1)
            tgt_mask = net.get_tgt_mask(sequence_length, device)
            
            optimizer.zero_grad()
            out = net(sequences, targets_input, tgt_mask =tgt_mask)
            #out=out.squeeze()
            loss=criterion(sequences, targets_expected, out)
            tot_loss+=loss.item()
            tot_items+=len(targets)
            loss.backward()
            if torch.isnan(loss):
                raise RuntimeError("NAN!")
            optimizer.step()
        tot_loss/=tot_items
        accuracy, tot_valid_loss = compute_accuracy(device, net, validloader, valid_criterion, "VALID", 
                                                    verbose = False, cv = True, correct_label_index=correct_label_index) #17th/33+1
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
            
        if round(accuracy,4)<=round(greatest_acc,4):
            if early_stop:
                j+=1
                if j>=early_stop and i>150:
                    break
        else:
            if accuracy>greatest_acc_overall:
                torch.save(net.state_dict(), model_path)
                greatest_acc_overall=accuracy
            j=0
            greatest_acc=accuracy
    print("Greatest accuracy on the run: {}".format(greatest_acc))
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
        
        
def get_class_weights(device, dataloader, indices_of_interest, norm_type):
    total_amount=0
    class_amounts=collections.Counter()
    bases = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    for sequences, labels in dataloader:
        values_of_interest = sequences[:, 16, indices_of_interest]
        _, original_bases = torch.max(values_of_interest, dim=1)

        values_of_interest = labels[:, correct_label_index, indices_of_interest] #
        _, new_bases = torch.max(values_of_interest, dim=1)
        for new_base, original_base in zip(new_bases, original_bases):
            correct_class = classes[bases[original_base.item()]+bases[new_base.item()]]
            class_amounts[correct_class]+=1
            total_amount+=1
            
    return create_class_weights(class_amounts, total_amount, norm_type)


if cv:
    if not os.path.isfile(last_run_log_file):
        with open(last_run_log_file, 'w+') as fw:
            fw.write("Run log.\n\n")
    k_folds=5
    epochs=5000
    seq_len=33
    kfold = KFold(n_splits=k_folds, shuffle=True) #batch size affects the size of datasets
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    for i in range(150): #Test with 150 different hyperparameter combinations
        valid_accuracies = list()
        learning_rate=random.sample([0.0001, 0.00001, 0.000001], 1)[0]
        lr_text=str(learning_rate).replace(".","d")
        train_batch_size=random.sample([32, 64, 128, 256], 1)[0]
        norm_type = random.sample(["sum", "max", "None"], 1)[0]
        
        nhead=random.sample([2, 3, 4, 6, 7, 12, 21, 42], 1)[0]
        num_encoder_layers=random.sample([2, 3, 4, 6, 8], 1)[0]
        num_decoder_layers=num_encoder_layers
        
        optimizer_type=random.sample(["Adam","AdamW"], 1)[0] #random.sample(["Adam","SGD"], 1)[0]
        weight_decay=random.sample([0, 0.000001, 0.00000001], 1)[0]
        if weight_decay!=0: weight_decay = round(weight_decay, -int(floor(log10(weight_decay))) + 2)
        decay_text="_wdecay"+str(weight_decay).replace(".","d")
        for (train_ids, test_ids) in kfold.split(combined_valid_train):
            
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            trainloader = torch.utils.data.DataLoader(
                          combined_valid_train, 
                          batch_size=train_batch_size, sampler=train_subsampler)
            validloader = torch.utils.data.DataLoader(
                              combined_valid_train,
                              batch_size=valid_batch_size, sampler=test_subsampler)
            train_class_weights = get_class_weights(device, trainloader, indices_of_interest, norm_type)
            valid_class_weights = get_class_weights(device, validloader, indices_of_interest, norm_type)
            
            net = Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers).to(device)
            if optimizer_type=="Adam":
                optimizer=torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            elif optimizer_type=="AdamW":
                optimizer=torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                raise RuntimeError("WRONG OPTIMIZER: {}".format(optimizer_type))
            criterion = WeightedMSELoss(device, train_class_weights, classes, correct_label_index, indices_of_interest)
            valid_criterion=WeightedMSELoss(device, valid_class_weights, classes, correct_label_index, indices_of_interest)
            run_name = None
            valid_acc = cv_train_network(net,criterion,valid_criterion,epochs,optimizer,trainloader, validloader, 
                                      correct_label_index, wandb=None)
            valid_accuracies.append(valid_acc)
        avg_valid_acc = sum(valid_accuracies) / len(valid_accuracies)

        print("Optimizer: {}\nLearning Rate: {}".format(optimizer_type, learning_rate))
        print("Weight decay: {}".format(weight_decay))
        print("Average validation accuracy: {}".format(avg_valid_acc))
        for k, accuracy in enumerate(valid_accuracies):
            print("Fold {}: {}.".format(k+1, accuracy))

        with open(last_run_log_file, 'a+') as fw:
            fw.write("Time: {}\n".format(datetime.now().strftime("%d.%m.%Y %H:%M:%S")))
            fw.write("Average validation accuracy: {}\n".format(avg_valid_acc))
            fw.write("\n".join(["Fold {}: {}.".format(k+1, accuracy) for k, accuracy in enumerate(valid_accuracies)]))
            fw.write("\nOptimizer: {}\nLearning Rate: {}\n".format(optimizer_type, learning_rate))
            fw.write("Weight decay: {}\nBatch size: {}\n".format(weight_decay, train_batch_size))
            fw.write("head: {}\nnum_encoder_layers: {}\nnum_decoder_layers: {}\n".format(nhead, num_encoder_layers, num_decoder_layers))
            fw.write("Norm type: {}\n".format(norm_type))
            fw.write("Target mode: {}\n".format(target_mode))
            fw.write('\n\n')

        if avg_valid_acc>greatest_avg_valid_acc:
            now = datetime.now()
            dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
            write_to_log_file(learning_rate, optimizer_type, weight_decay,
                             train_batch_size, valid_batch_size, cv_log_file, model_version, 
                              input_to_matrix, 
                              avg_valid_acc, model_path, many_classes, compute_accuracy_file,
                              model_script, run_name, dt_string, target_mode, 
                              nhead, num_encoder_layers, num_decoder_layers, acc_list=valid_accuracies, norm=norm,
                                 norm_type=norm_type)
            greatest_avg_valid_acc=avg_valid_acc
            os.path.join("saved_models", "transformer", "test_notebook.pth")
            torch.save(net.state_dict(), model_path)
            

greatest_acc_overall=0
if os.path.isfile(log_file):
    greatest_acc_overall=get_earlier_accuracy(log_file)
print(greatest_acc_overall)
train_batch_size=32

trainloader = torch.utils.data.DataLoader(train_dataset
    ,batch_size=train_batch_size
    ,shuffle=True
    ,drop_last=True
)
validloader = torch.utils.data.DataLoader(valid_dataset
    ,batch_size=5
    ,shuffle=False
)

if train:
    learning_rate=0.0001
    optimizer_type="AdamW"
    weight_decay=0.000001
    nhead=42
    num_encoder_layers=6
    num_decoder_layers=6
    epochs=5000
    norm_type="None"
    criterion = WeightedMSELoss(device, class_weights_train_whole, classes, correct_label_index, indices_of_interest)
    valid_criterion=WeightedMSELoss(device, class_weights_valid_whole, classes, correct_label_index, indices_of_interest)
    lr_text=str(learning_rate).replace(".","d")
    decay_text="_wdecay"+str(weight_decay).replace(".","d")
    for i in range(1000):
        net = Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers).to(device)
        if optimizer_type=="Adam":
            optimizer=torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type=="AdamW":
            optimizer=torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise RuntimeError("WRONG OPTIMIZER.")
        run_name=None
#         run = wandb.init(project=project_name)
#         run_name="run_{}".format(i)
#         wandb.run.name = run_name
#         config = wandb.config
#         config.batch_size=train_batch_size
            
            
        greatest_acc = train_network(net,criterion, valid_criterion,epochs,optimizer,trainloader, validloader,
                          correct_label_index, greatest_acc_overall, model_path, wandb=None, early_stop=100)
        
        
        print("Accuracy: {}".format(greatest_acc))
        
        with open(last_run_log_file_final_runs, 'a+') as fw:
            fw.write("Time: {}\n".format(datetime.now().strftime("%d.%m.%Y %H:%M:%S")))
            fw.write("Validation accuracy: {}\n".format(greatest_acc))
            fw.write("\nOptimizer: {}\nLearning Rate: {}\n".format(optimizer_type, learning_rate))
            fw.write("Weight decay: {}\nBatch size: {}\n".format(weight_decay, train_batch_size))
            fw.write("head: {}\nnum_encoder_layers: {}\nnum_decoder_layers: {}\n".format(nhead, num_encoder_layers, num_decoder_layers))
            fw.write("Norm type: {}\n".format(norm_type))
            fw.write("Target mode: {}\n".format(target_mode))
            fw.write('\n\n')
       
        
        if greatest_acc>greatest_acc_overall:
            now = datetime.now()
            dt_string = now.strftime("%d.%m.%Y %H:%M:%S")
            write_to_log_file(learning_rate, optimizer_type, weight_decay,
                             train_batch_size, valid_batch_size, log_file, model_version, 
                              input_to_matrix, 
                              greatest_acc, model_path, many_classes, compute_accuracy_file,
                              model_script, run_name, dt_string, target_mode, 
                              nhead, num_encoder_layers, num_decoder_layers, norm=norm,
                                 norm_type=norm_type)
            greatest_acc_overall=greatest_acc
            
            
testloader = torch.utils.data.DataLoader(test_dataset
    ,batch_size=5
    ,shuffle=False
)

nhead=42
num_encoder_layers=6
num_decoder_layers=6

if data_mode=="transformer": #Trained more models after the chosen model
    model_path=os.path.join("saved_models", data_mode, "transformer_v10_masters_thesis.pth")
model=Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
#model.load_state_dict(torch.load(model_path))
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

criterion=WeightedMSELoss(device, class_weights_test, classes, correct_label_index, indices_of_interest)
accuracy, tot_valid_loss = compute_accuracy(device, model, testloader, None, "TEST", 
                                                    verbose = True, cv = False, correct_label_index=correct_label_index)
print("Test accuracy:",accuracy)


accuracy, tot_valid_loss = compute_accuracy(device, model, validloader, None, "VALID", 
                                                    verbose = True, cv = False, correct_label_index=correct_label_index)
print("Valid accuracy:",accuracy)