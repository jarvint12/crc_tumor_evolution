# crc_tumor_evolution
Master's thesis for predicting tumor evolution in colorectal cancer

Data folder contains all the data utilized in my thesis, but they have to be unzipped before use.

Figures folder contains generated figures for the thesis. Similarly, roc_curves folder contains generated ROC curves.

model_skeletons folder contains all the different architectures, which were tested during my master's thesis.

logs-folder contains results for different architectures and datatypes. Logs contain all the important information about the training and the accuracies.

process_inputs folder contains scripts, that are used to read data files to matrices and dataloaders.

saved_models contains the best performed models saved as .pth files. Thus, you can load the models using the architectures illustrated in log files.

For every model, there is a .ipynb file and a .py file, which either can be used to use the models. The models and scripts are paired as follow:

<b>MLP</b>: basemodel_fc.ipynb, fc_v6.py

<b>CNN_BASE</b>: basemodel_cnn.ipynb, cnn_base_clu4.py

<b>CNN</b>: CNN_pipeline.ipynb, cnnv16_clu4.py

<b>RNN</b>: RNN_pipeline.ipynb, rnnv6_clu4.py

<b>Transformer</b>: transformer_pipeline.ipynb, transformer_whole_matrix4.py

