import os
import argparse

def get_all_metrics(input_fold, prefix, excludes, suffixes=[]):
    file_metrics=list()
    for root, _, files in os.walk(input_fold):
        for file in files:
            if not file.startswith(prefix):
                continue
            elif any([exclude in file for exclude in excludes]):
                continue
            elif suffixes and (not any([file.endswith(suffix) for suffix in suffixes])):
                continue
            accuracies=list()
            file_path=os.path.join(root, file)
            with open(file_path, 'r') as fr:
                for line in fr:
                    if line.startswith("Average validation accuracy: ") or "Average validation accuracy:" in line or line.startswith("Accuracy: ") or line.startswith("Validation accuracy: "):
                        accuracies.append(float(line.strip().split(' ')[-1]))
            if accuracies:
                file_metrics.append((file_path.split('/')[-1], sum(accuracies)/len(accuracies), len(accuracies), max(accuracies)))
            else:
                file_metrics.append((file_path.split('/')[-1],0,0,0))

    for file_tuple in sorted(file_metrics, key=lambda x: x[-1], reverse=True):
        print(file_tuple)

def main(mode, data_type):
    assert mode or data_type, "Give model parameter or data type"
    if mode=="mlp":
        input_fold="huslab_timo_dev/masters_thesis/logs/basemodel_fc/"
        #raise RuntimeError("CHECK SCRIPT")
        get_all_metrics(input_fold, "cv_base_fc_", excludes=[], suffixes=[".log"])
        print("-----------------------------")
        get_all_metrics(input_fold, prefix="train_base_fc_", excludes=[], suffixes=["_all_runs.log"])
    elif mode=="basemodel_cnn":
        input_fold="huslab_timo_dev/masters_thesis/logs/basemodel_cnn/"
        #raise RuntimeError("CHECK SCRIPT")
        get_all_metrics(input_fold, "current_run_log_v", excludes=[], suffixes=[".log"])
        print("-----------------------------")
        get_all_metrics(input_fold, prefix="base_cnn_v", excludes=[], suffixes=[".log"])
    elif mode=="cnn":
        input_fold="huslab_timo_dev/masters_thesis/logs/cnn/"
        #get_all_metrics(input_fold, "cv_cnn", excludes=[], suffix="_old.log")
        get_all_metrics(input_fold, "current_run_log_cnn_v1", ["old"])
        print("-----------------------------")
        get_all_metrics(input_fold, prefix="cnn_v1", excludes=["old", "cheat"], suffixes=["all_runs.log"])
    elif mode=="rnn":
        input_fold="huslab_timo_dev/masters_thesis/logs/rnn/"
        get_all_metrics(input_fold, "current_run_log_", excludes=[])
        print("-----------------------------")
        get_all_metrics(input_fold, prefix="rnn_v1_", excludes=[], suffixes=["_all_runs.log"])
    elif mode=="transformer":
        input_fold="huslab_timo_dev/masters_thesis/logs/transformer/"
        get_all_metrics(input_fold, "transformer_", excludes=[], suffixes=["_cv_all_runs.log", "_cv_all_runs_modloss.log"])
        print("-----------------------------")
        get_all_metrics(input_fold, prefix="transformer_", excludes=["_cv_"], suffixes=["_all_runs.log"])
    else:
        input_fold=os.path.join("huslab_timo_dev/masters_thesis/logs",data_type)
        get_all_metrics(input_fold, "", excludes=[], suffixes=["_all_runs.log"])
        
        
def optparsing():
    parser = argparse.ArgumentParser(description="Your program description here")

    # Add the argument for choosing the model
    parser.add_argument(
        "--model",
        choices=["mlp", "basemodel_cnn", "cnn", "rnn", "transformer"],
        help="Choose a model type (mlp, basemodel_cnn, cnn, rnn, transformer)",
    )
    parser.add_argument(
        "--data_type",
        choices=["clu3", "clu4", "clu431"],
        help="If you want to count performance for all datatypes",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = optparsing()
    main(args.model, args.data_type)