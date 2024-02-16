import torch

def count_f_scores(tp,fp,tn,fn):
    if (tp+fp)==0:
        precision=0
    else:
        precision=tp/(tp+fp)
    if tp+fn==0:
        recall=0
    else:
        recall=tp/(tp+fn)
    if precision==0 and recall==0:
        f1=f2=0
    else:
        f1=2*(precision*recall)/(precision+recall)
        f2=5*(precision*recall)/(4*precision+recall)
    return f1, f2, precision, recall


def count_pred_types(ftp, ftn, tp,fp,tn,fn, labels, predicted, original_bases):
    for original_base, label, prediction in zip(original_bases, labels, predicted):
        if original_base==label:
            if label==prediction:
                tn+=1
            else:
                fp+=1
                #if prediction in [1,7]:
                 #   ftn+=1
        else:
            if label==prediction:
                tp+=1
            else:
                fn+=1
                if prediction!=original_base:
                #if prediction not in [1,7]:
                    ftp+=1
    return tp,fp,tn,fn,ftn, ftp

def compute_accuracy(device, net, dataloader, criterion, datatype, verbose, cv, correct_label_index):
    net.eval()
    bases = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    classes={'CA': 0, 'CC': 1, 'CG': 2, 'CT': 3, 'TA': 4, 'TC': 5, 'TG': 6, 'TT': 7}
    correct = 0
    tp=fp=tn=fn=ftn=ftp=0
    number_of_classes=len(classes)
    correct_per_class=[0 for i in range(number_of_classes)]
    total_per_class=[0 for i in range(number_of_classes)]
    with torch.no_grad():
        tot_loss=0
        tot_items=0
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            labels_input = labels[:,:-1]
            labels_expected = labels[:,1:]
                        
            sequence_length = labels_input.size(1)
            tgt_mask = net.get_tgt_mask(sequence_length, device)
                        
            output = net(sequences, labels_input, tgt_mask=tgt_mask)
            indices_of_interest = [80, 81, 82, 83]
            
            #Get the predicted bases, output should not contain SOS token? Thus, the correct base in 16th/0th position
            values_of_interest = output[:, correct_label_index, indices_of_interest] #Start token removed
            max_values, results = torch.max(values_of_interest, dim=1)
            
            #Get the expected bases, shifted right, so does not contain SOS token. Thus, the correct base in 16th/0th position
            values_of_interest = labels_expected[:, correct_label_index, indices_of_interest] #
            correct_max_values, correct_results = torch.max(values_of_interest, dim=1)
            
            #Get the input base. Input does not contain SOS token, and is always 33 bases long. Thus, the correct base in 16th position
            values_of_interest = sequences[:, 16, indices_of_interest]
            _, original_bases = torch.max(values_of_interest, dim=1)
            
            for result, correct_result, original_base in zip(results, correct_results, original_bases):
                correct_class = classes[bases[original_base.item()]+bases[correct_result.item()]]
                correct_per_class[correct_class]+=(correct_result==result).item()
                total_per_class[correct_class]+=1

            if cv or verbose:
                tot_items+=len(labels)
            if cv:
                tot_loss+=criterion(sequences, labels_expected, output).item()
            if verbose:
                tp,fp,tn,fn, ftn, ftp = count_pred_types(ftp, ftn, tp,fp,tn,fn, correct_results, results, original_bases)
        if cv:
            tot_loss/=tot_items
        for i in range(number_of_classes):
            correct+=(correct_per_class[i]/total_per_class[i])/number_of_classes
        if verbose:
            f1, f2, precision, recall = count_f_scores(tp,fp,tn,fn)
            f1_fake, f2_fake, fake_precision, fake_recall = count_f_scores((tp+ftp),(fp-ftn),(tn+ftn),(fn-ftp))
            tn_tnfp=tn/(tn+fp) if tn+fp>0 else 0
            fake_tpftp_tpftpfn=(tp+ftp)/(tp+ftp+fn) if (tp+ftp+fn)>0 else 0
            fake_tnftn_tnftnfp=(tn+ftn)/(tn+ftn+fp) if (tn+ftn+fp)>0 else 0
            print('\n',datatype)
            print("TP:",tp,". FN:",fn, "TP/(TP+FN):",recall,"TN:",tn,"FP:",fp,"TN/(TN+FP):",tn_tnfp,
                  "Wrong positive class predicted:",ftp, "Wrong negative class predicted:",ftn)
            print("Fake F1-score:",f1_fake,". Fake F2-score:",f2_fake)
            print("Fake TP/(TP+FN):",fake_tpftp_tpftpfn,"Fake TN/(TN+FP)",fake_tnftn_tnftnfp)
            print("Fake precision:",fake_precision,"Fake recall:",fake_recall)
            print("F1-score:",f1)
            print("F2-score:",f2)
            print("Precision:",precision)
            print("Recall:",recall)
            print("Fake accuracy:",(tn+ftn+tp+ftp)/tot_items)

    return correct, tot_loss