import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import os

norm=True #True #None/False
    
from process_input.input_to_2Dmatrix_transformer_classes import parse_matrices as parse_matrices_fc, FeatureDataset as FeatureDataset_fc
from model_skeletons.fc_basemodel.base_fc_v6 import base_fc, model_version as model_version_fc

from process_input.input_to_1D_seq_matrix import parse_matrices as parse_matrices_cnn_base, FeatureDataset as FeatureDataset_cnn_base
from model_skeletons.cnn_basemodel.base_cnn_v5 import base_cnn, model_version as model_version_cnn_base

from model_skeletons.cnn.cnn_v1_16 import cnn, model_version as model_version_cnn
from process_input.input_to_3Dmatrix import parse_matrices as parse_matrices_cnn, FeatureDataset as FeatureDataset_cnn

from process_input.input_to_2Dmatrix import parse_matrices as parse_matrices_rnn, FeatureDataset as FeatureDataset_rnn
from model_skeletons.rnn.rnn_v1_6 import rnn, model_version as model_version_rnn

from process_input.input_to_2Dmatrix_transformer import parse_matrices as parse_matrices_transformer, FeatureDataset as FeatureDataset_transformer
from model_skeletons.transformer.transformer_v10 import Transformer, model_version as model_version_transformer
correct_label_index = 16
target_mode = "whole_matrix"
norm_type = "None"
    



def plot_roc_curve(destination, plot_data, title):
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    
    labels = ["MLP", "cnn_base", "cnn", "rnn", "transformer"]
    line_colors = ['#1b9e77', '#d95f02', '#7570b3', '#66a61e', '#e7298a']
    
    for index, (fpr, tpr) in enumerate(plot_data):
        plt.plot(fpr, tpr, color=line_colors[index], lw=2, label=labels[index])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    # Save the figure to a file
    plt.savefig(os.path.join(destination, 'roc_curve_combined.png'))


def get_results_transformer(device, testloader, model):
    all_results = []
    all_new_labels = []
    indices_of_interest = [80, 81, 82, 83]
    with torch.no_grad():
        for sequences, _, targets in testloader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            targets_input = targets[:,:-1]
            targets_expected = targets[:,1:]
                        
            sequence_length = targets_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length, device)

            output = model(sequences, targets_input, tgt_mask=tgt_mask)
            
            #Get the predicted bases, output should not contain SOS token? Thus, the correct base in 16th/0th position
            values_of_interest = output[:, correct_label_index, indices_of_interest] #Start token removed
            probabilities = F.softmax(values_of_interest, dim=1)
            
            #Get the expected bases, shifted right, so does not contain SOS token. Thus, the correct base in 16th/0th position
            values_of_interest = targets_expected[:, correct_label_index, indices_of_interest] #
            _, correct_results = torch.max(values_of_interest, dim=1)
            
            #Get the input base. Input does not contain SOS token, and is always 33 bases long. Thus, the correct base in 16th position
            values_of_interest = sequences[:, 16, indices_of_interest]
            _, original_bases = torch.max(values_of_interest, dim=1)

            for index, (correct_base, original_base) in enumerate(zip(correct_results, original_bases)):
                if correct_base==original_base:
                    all_new_labels.append(torch.tensor(0))
                else:
                    all_new_labels.append(torch.tensor(1))
                result = 1 - (probabilities[index,original_base.item()]) #1-p(base_does_not_mutate)
                all_results.append(result.detach().cpu().numpy())
    all_results = [arr.reshape(1, -1) if arr.ndim == 0 else arr for arr in all_results]
    all_new_labels = [arr.reshape(1, -1) if arr.ndim == 0 else arr for arr in all_new_labels]
    return all_results, all_new_labels


def get_results_8_classes_mlp(device, testloader, model):
    all_results = []
    all_new_labels = []

    for sequences, targets, mut_landscape, labels in testloader:
        sequences, targets, mut_landscape, labels = sequences.to(device), targets.to(device), mut_landscape.to(device), labels.to(device)
        output=model(sequences, mut_landscape)

        # Combine second and last columns for the negative class

        probabilities = F.softmax(output, dim=1)

        # Extract probabilities for classes 1 and 7
        p_class_1 = probabilities[:, 1]
        p_class_7 = probabilities[:, 7]

        # Get the probability that base mutates
        result = 1 - (p_class_1 + p_class_7)

        # Add label 0, if class is negative ('CC' or 'TT') else 1
        new_labels = torch.where((labels == 1) | (labels == 7), torch.tensor(0), torch.tensor(1))
        
        # Append results and new_labels to the lists
        all_results.append(result.detach().cpu().numpy())
        all_new_labels.append(new_labels.cpu().numpy())
    return all_results, all_new_labels



def get_results_8_classes(device, testloader, model):
    all_results = []
    all_new_labels = []

    for sequences, labels in testloader:
        sequences, labels = sequences.to(device), labels.to(device)
        output=model(sequences)
        
        if isinstance(output, tuple): #RNN case returns also hidden state
            output, _ = output
        else:
            output = output
        # Combine second and last columns for the negative class

        probabilities = F.softmax(output, dim=1)

        # Extract probabilities for classes 1 and 7
        p_class_1 = probabilities[:, 1]
        p_class_7 = probabilities[:, 7]

        # Get the probability that base mutates
        result = 1 - (p_class_1 + p_class_7)

        # Add label 0, if class is negative ('CC' or 'TT') else 1
        new_labels = torch.where((labels == 1) | (labels == 7), torch.tensor(0), torch.tensor(1))
        
        # Append results and new_labels to the lists
        all_results.append(result.detach().cpu().numpy())
        all_new_labels.append(new_labels.cpu().numpy())
    return all_results, all_new_labels



def get_transformer_data(test_file, norm, target_mode, norm_type, parse_matrices_function, FeatureDataset_func):
    test_input, test_target, class_weights_test, labels_test = parse_matrices_function(test_file, norm, target_mode, norm_type)
            
    sos_token = torch.zeros((1, 1, test_input.shape[2]))
    eos_token = torch.ones((1, 1, test_input.shape[2]))

    test_input_with_tokens = torch.cat([test_input, eos_token.expand(test_input.size(0), 1, -1)], dim=1)
    test_target_with_tokens = torch.cat([sos_token.expand(test_input.size(0), 1, -1), test_target, eos_token.expand(test_input.size(0), 1, -1)], dim=1)
    test_dataset=FeatureDataset_func(data=test_input_with_tokens, labels=labels_test, targets=test_target_with_tokens)
    
    return test_dataset


    
def create_inputs(destination, device, parse_matrices_functions, FeatureDataset_funcs,
                     models, test_files, 
                        classes, norm, title):
    
    plot_data=list()
    for index, (model, parse_matrices_function, FeatureDataset_func) in enumerate(zip(models, parse_matrices_functions, FeatureDataset_funcs)):
        if index==1: #cnn_basemodel:
            test_input, test_target = parse_matrices_function(test_files[0], classes)
        elif index==2 or index==3: #CNN or RNN
            test_input, test_target = parse_matrices_function(test_files[1], classes, norm)
        if index==0: #MLP
            test_input, test_target, test_mut_landscape, test_labels = parse_matrices_function(test_files[1], norm)
            test_dataset=FeatureDataset_func(data=test_input, targets=test_target, mut_landscape =
                            test_mut_landscape, labels = test_labels)
        elif index==4: #Transformer
            test_dataset = get_transformer_data(test_files[1], norm, target_mode, norm_type, parse_matrices_function, FeatureDataset_func)
        else:
            test_dataset=FeatureDataset_func(data=test_input, labels=test_target)
        testloader = DataLoader(test_dataset
                                ,batch_size=5
                                    ,shuffle=False
                                )

        if index==0:
            all_results, all_new_labels = get_results_8_classes_mlp(device, testloader, model)
        elif index==4:
            all_results, all_new_labels = get_results_transformer(device, testloader, model)
        else:
            all_results, all_new_labels = get_results_8_classes(device, testloader, model)
    
        
        # Concatenate the collected tensors into one large matrix along the batch dimension
        all_results = np.concatenate(all_results, axis=0)
        all_new_labels = np.concatenate(all_new_labels, axis=0)
        
        #Calculate ROC metrics
        fpr, tpr, _ = roc_curve(all_new_labels, all_results)

        plot_data.append((fpr, tpr))
    
    plot_roc_curve(destination, plot_data, title)


def get_model(device, model_path, model):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model

def main():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device('cuda')
    classes={'CA': 0, 'CC': 1, 'CG': 2, 'CT': 3, 'TA': 4, 'TC': 5, 'TG': 6, 'TT': 7}
    destination="{}/roc_curves".format(current_script_dir)
    title="F"
    test_files=["{}/data/20220214/sompred_crc9_clu9_pyri_mut_predict.matrix".format(current_script_dir),
                "{}/data/20220214/sompred_crc9_clu1_pyri_mut_combined_test.matrix".format(current_script_dir)]
    models = list()
    parse_matrices_functions = list()
    FeatureDataset_funcs = list()
    
    model_path="{}/saved_models/basemodel_fc/{}.pth".format(current_script_dir, model_version_fc)
    model = base_fc()
    models.append(get_model(device, model_path, model))
    parse_matrices_functions.append(parse_matrices_fc)
    FeatureDataset_funcs.append(FeatureDataset_fc)

    model_path="{}/saved_models/basemodel_cnn/{}.pth".format(current_script_dir, model_version_cnn_base)
    model = base_cnn(many_classes=True)
    models.append(get_model(device, model_path, model))
    parse_matrices_functions.append(parse_matrices_cnn_base)
    FeatureDataset_funcs.append(FeatureDataset_cnn_base)
        

    model_path="{}/saved_models/cnn/{}.pth".format(current_script_dir, model_version_cnn)
    model = cnn()
    models.append(get_model(device, model_path, model))
    parse_matrices_functions.append(parse_matrices_cnn)
    FeatureDataset_funcs.append(FeatureDataset_cnn)

    model_path="{}/saved_models/rnn/{}.pth".format(current_script_dir, model_version_rnn)
    model = rnn(bidirectional=False, hidden_dim=22, n_layers=1)
    models.append(get_model(device, model_path, model))
    parse_matrices_functions.append(parse_matrices_rnn)
    FeatureDataset_funcs.append(FeatureDataset_rnn)

    model_path="{}/saved_models/transformer/{}_masters_thesis.pth".format(current_script_dir, model_version_transformer)
    nhead=42
    num_encoder_layers=6
    num_decoder_layers=6
    model=Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
    models.append(get_model(device, model_path, model))
    parse_matrices_functions.append(parse_matrices_transformer)
    FeatureDataset_funcs.append(FeatureDataset_transformer)
    
    create_inputs(destination, device, parse_matrices_functions, FeatureDataset_funcs,
                     models, test_files, classes, norm, title)

if __name__=='__main__':
    main()