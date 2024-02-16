import random
import os

def new_files(orig_files, classes, training_file_path, validation_file_path, test_file_path=None):
    class_lines=[dict() for _ in range(len(classes))]
    ids=[list() for i in range(len(classes))]
    lines_to_write=list()
    amount_of_ids = 0
    for file in orig_files:
        lines_to_write=list()
        row=0
        with open(file, 'r') as fr:
            for line in fr:
                lines_to_write.append(line)
                if line.startswith('#'):
                    if row>0:
                        row=0
                    if line.startswith('#ID:'):
                        line_id=line.strip()
                    if line.startswith('#REF:'):
                        ref=line[6]
                    elif line.startswith('#ALT:'):
                        alt=line[6]
                        current_class=classes[ref+alt]
                    continue
                row+=1
                if row==4:
                    if not line_id in ids[current_class]:
                        amount_of_ids+=1
                        ids[current_class].append(line_id)
                        class_lines[current_class][line_id]=list()
                    class_lines[current_class][line_id].append(lines_to_write)
                    lines_to_write=list()
    print("Amount of IDS:",amount_of_ids)
    if test_file_path==None:
        for index in range(len(classes)):
            number_of_ids=len(ids[index])
            val_ids=random.sample(ids[index], int(0.15*number_of_ids))
            with open(validation_file_path, 'a+') as fw:
                for val_id in val_ids:
                    for line_chunk in class_lines[index][val_id]:
                        for line in line_chunk:
                            fw.write(line)
                    del class_lines[index][val_id]
            with open(training_file_path, 'a+') as fw:
                for single_id in ids[index]:
                    if single_id in val_ids:
                        continue
                    for line_chunk in class_lines[index][single_id]:
                        for line in line_chunk:
                            fw.write(line)
    else:
        for index in range(len(classes)):
            with open(test_file_path, 'a+') as fw:
                for single_id in ids[index]:
                    for line_chunk in class_lines[index][single_id]:
                        for line in line_chunk:
                            fw.write(line)
                            
                            
def check_if_output_exists(file):
    if os.path.isfile(file):
        while True:
            overwrite = input("File '{}' already exists. Want to overwrite?\n".format(file))
            if overwrite.lower() in ["yes", "y"]:
                os.remove(file)
                return False
            elif overwrite.lower() in ["no", "n"]:
                print("Exiting...")
                return True
    return False
                            
def main():
    generate_test_files = False
    data_mode = "clu4"
    classes={'CA': 0, 'CC': 1, 'CG': 2, 'CT': 3, 'TA': 4, 'TC': 5, 'TG': 6, 'TT': 7}
    orig_train_files=["data/{0}/sompred_crc9_{0}_pyri_mut_train.ccle.matrix".format(data_mode),
                      "data/{0}/sompred_crc9_{0}_pyri_mut_train.cosmic.matrix".format(data_mode),
                      "data/{0}/sompred_crc9_{0}_pyri_mut_train.matrix".format(data_mode)
                     ]
    train_file=orig_train_files[2].replace("mut_train.matrix", "mut_combined_train.matrix")
    if check_if_output_exists(train_file):
        quit()
    valid_file=orig_train_files[2].replace("mut_train.matrix", "mut_combined_valid.matrix")
    new_files(orig_train_files, classes, train_file, valid_file)
    
    if generate_test_files:
        orig_test_files=["data/20220214/sompred_crc9_clu9_pyri_mut_predict.ccle.matrix",
              "data/20220214/sompred_crc9_clu9_pyri_mut_predict.cosmic.matrix",
              "data/20220214/sompred_crc9_clu9_pyri_mut_predict.matrix"
                ]
        test_file="data/20220214/sompred_crc9_clu1_pyri_mut_combined_test.matrix"
        if check_if_output_exists(test_file):
            quit()
        new_files(orig_test_files, classes, None, None, test_file)

if __name__=='__main__':
    main()