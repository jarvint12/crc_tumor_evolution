import torch

def count_f_scores(tp,fp,fn):
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

def count_pred_types(ftp, ftn, tp,fp,tn,fn, labels, predicted):
    for label, prediction in zip(labels, predicted):
        if label in [1,7]:
            if label==prediction:
                tn+=1
            else:
                fp+=1
                if prediction in [1,7]:
                    ftn+=1
        else:
            if label==prediction:
                tp+=1
            else:
                fn+=1
                if prediction not in [1,7]:
                    ftp+=1
    return tp,fp,tn,fn,ftn, ftp


def get_class_name(acc_dict, index):
    for key, value in acc_dict.items():
        if value == index:
            return key
        
        
def compute_accuracy(device, net, dataloader, criterion, datatype, model_type, verbose, cv):
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
        for sequences, targets, mut_landscape, labels in dataloader:
            sequences, targets, mut_landscape, labels = sequences.to(device), targets.to(device), mut_landscape.to(device), labels.to(device)
            
            if model_type == "rnn":
                outputs, _ = net(sequences, mut_landscape)
            else:
                outputs = net(sequences, mut_landscape)

            _, predicted = torch.max(outputs.data, 1)

            for index, (label, prediction) in enumerate(zip(labels, predicted)):
                correct_per_class[int(label.item())]+=(label.item()==prediction.item())
                total_per_class[int(label.item())]+=1

            if cv or verbose:
                tot_items+=len(labels)
            if cv:
                tot_loss+=criterion(outputs,labels).item()
            if verbose:
                tp,fp,tn,fn, ftn, ftp = count_pred_types(ftp, ftn, tp,fp,tn,fn, labels, predicted)
        if cv:
            tot_loss/=tot_items
        for i in range(number_of_classes):
            correct+=(correct_per_class[i]/total_per_class[i])/number_of_classes
            #if verbose:
#                 print("Class {}: Correct {}, total {}, acc {}".format(get_class_name(classes, i), correct_per_class[i], total_per_class[i], correct_per_class[i]/total_per_class[i]))

        if verbose:
            f1, f2, precision, recall = count_f_scores(tp,fp,fn)
            f1_fake, f2_fake, fake_precision, fake_recall = count_f_scores((tp+ftp),(fp-ftn),(fn-ftp))
            tn_tnfp=tn/(tn+fp) if tn+fp>0 else 0
            fake_tpftp_tpftpfn=(tp+ftp)/(tp+ftp+fn) if (tp+ftp+fn)>0 else 0
            fake_tnftn_tnftnfp=(tn+ftn)/(tn+ftn+fp) if (tn+ftn+fp)>0 else 0
            print('\n{}'.format(datatype))
            print("TP:",tp,". FN:",fn, "TP/(TP+FN):",recall,"TN:",tn,"FP:",fp,"TN/(TN+FP):",tn_tnfp,
                  "Wrong positive class predicted:",ftp, "Wrong negative class predicted:",ftn)
            print("Fake F1-score:",f1_fake,". Fake F2-score:",f2_fake)
            print("Fake TP/(TP+FN):",fake_tpftp_tpftpfn,"Fake TN/(TN+FP)",fake_tnftn_tnftnfp)
            print("Fake precision:",fake_precision,"Fake recall:",fake_recall)
            print("F1-score:",f1)
            print("F2-score:",f2)
            print("Precision:",precision)
            print("Recall:",recall)
            print("Fake accuracy:",0.5*(tn+ftn)/(tn+fp)+0.5*(tp+ftp)/(tp+fn))#(tn+ftn+tp+ftp)/tot_items)

    return correct, tot_loss