import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as fm


def plot_amounts_against_data_size(destination, data, fig_name, color_map, acc_col_index, dataset_type):
    
    plt.figure()
    legend_font_size = 12.5
    
    # Amount of data for each class
    x_values = {'clu3': 1070, 'clu4': 4115, 'clu1': 10870, 'clu431': 16055}
    y_values = {}
    for entry in sorted(data, key=lambda x: x_values[x[0]]):
        if entry[1] not in y_values:
            y_values[entry[1]] = {'x': [], 'y': []}
        y_values[entry[1]]['x'].append(x_values[entry[0]])
        y_values[entry[1]]['y'].append(entry[acc_col_index])
    
    
    for model, values in y_values.items():
        plt.plot(values['x'], values['y'], label=model, color=color_map[model])

    plt.xlabel('Amount of samples')
    plt.ylabel('Test accuracy'.format(dataset_type))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Formatting y-axis ticks to 2 decimals
    plt.ylim(0.2, 1)
    plt.legend(prop=fm.FontProperties(size=legend_font_size), handlelength=2.5) # Set font size of the legend labels
    plt.savefig(os.path.join(destination, fig_name), dpi=600)

def plot_data(destination, data, fig_name, color_map, plot_location=None):
    
    plt.figure()
    
    marker_symbols = {'clu4': 'o', 'clu431': 's', 'clu3': '^', 'clu1': 'x'}
    marker_size = 10
    legend_font_size = 11
    
    # Create scatter plot
    for row in data:
        marker_type, color_type, x, y = row
        plt.scatter(x, y, marker=marker_symbols[marker_type], color=color_map[color_type], label=f'{marker_type}-{color_type}',
                  s=120)

    # Add labels and legend
    plt.xlabel('Validation accuracy')
    plt.ylabel('Test accuracy')
    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#1b9e77', markersize=marker_size, label='MLP', lw=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#d95f02', markersize=marker_size, label='Base CNN'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7570b3', markersize=marker_size, label='CNN'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#66a61e', markersize=marker_size, label='RNN'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e7298a', markersize=marker_size, label='Transformer'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='black', markersize=marker_size, label='ETP1', fillstyle='none'),
        Line2D([0], [0], marker='^', color='w', markeredgecolor='black', markersize=marker_size, label='ETP3', fillstyle='none'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='black', markersize=marker_size, label='ETP4', fillstyle='none'),
        Line2D([0], [0], marker='s', color='w', markeredgecolor='black', markersize=marker_size, label='ETP431', fillstyle='none')
    ]
    
    plt.xlim(0, 1)
    plt.ylim(0, 0.95)

    #plt.legend(handles=legend_elements, loc='upper right')
    if plot_location:
        legend = plt.legend(handles=legend_elements, loc=plot_location, prop=fm.FontProperties(size=legend_font_size), handlewidth=10, linewidth=10)
    else:
        legend = plt.legend(handles=legend_elements, prop=fm.FontProperties(size=legend_font_size), handlewidth=10, linewidth=10)
    for line in legend.get_lines():
        line.set_linewidth(10)  # Adjust the line width as needed

    plt.savefig(os.path.join(destination, fig_name), dpi=600)
    
def main():
    plot_acc_comparisons=False
    plot_amounts_vs_acc = True
    destination = "/u/77/jarvint12/unix/huslab_timo_dev/masters_thesis/figures"
    
    color_map = {'mlp': '#1b9e77', 'base_cnn': '#d95f02', 'cnn': '#7570b3', 'rnn': '#66a61e', 'transformer': '#e7298a'}
    
    data = [
        ['clu4', 'mlp', 0.41968558430001357, 0.32111093706488447],
        ['clu431', 'mlp', 0.3797064583230668, 0.39834261133603244],
        ['clu3', 'mlp', 0.5280446269416857, 0.3300346552813658],
        ['clu1', 'mlp', 0.34776595016457595, 0.3950067811087548],

        ['clu4', 'base_cnn', 0.46277237279758743, 0.32029158098237037],
        ['clu431', 'base_cnn', 0.5052516064615092, 0.43086247938221617],
        ['clu3', 'base_cnn', 0.7753472222222222, 0.25124459787288733],
        ['clu1', 'base_cnn', 0.6226074657217044, 0.40247420099393777],

        ['clu4', 'cnn', 0.9075065179958857, 0.3565519367328578],
        ['clu431', 'cnn', 0.4705683618841499, 0.39903377567851256],
        ['clu3', 'cnn', 0.5577325248281131, 0.24226146964633805],
        ['clu1', 'cnn', 0.477788344680632, 0.41197193705417384],

        ['clu4', 'rnn', 0.7974614931990642, 0.32631659276396113],
        ['clu431', 'rnn', 0.46816065239999843, 0.4095490127562495],
        ['clu3', 'rnn', 0.5439171122994653, 0.36880412168241117],
        ['clu1', 'rnn', 0.4713885746799314, 0.44946366985840674],

        ['clu4', 'transformer', 0.8123443429221859, 0.290913247327721],
        ['clu431', 'transformer', 0.4304695953485014, 0.5105582465779834],
        ['clu3', 'transformer', 0.36546266233766234, 0.28107377819548873],
        ['clu1', 'transformer', 0.4987097792505377, 0.4625576360773729],
    ]
    
    if plot_acc_comparisons:
        plot_data(destination, data, 'clu_accuracies.png', color_map, plot_location='upper left')
    if plot_amounts_vs_acc:
        plot_amounts_against_data_size(destination, data, 'real_test_accuracies_against_data_size.png', color_map, 3, 'Test')
        plot_amounts_against_data_size(destination, data, 'real_valid_accuracies_against_data_size.png', color_map, 2, 'Valid')
    
    
    data = [
        ['clu4', 'mlp', 0.6474248291508111, 0.6479591836734694],
        ['clu431', 'mlp', 0.6375410003100089, 0.6683673469387755],
        ['clu3', 'mlp', 0.587957974137931, 0.6377551020408163],
        ['clu1', 'mlp', 0.5976301414454379, 0.6377551020408163],

        ['clu4', 'base_cnn', 0.5322510822510823, 0.5408163265306123],
        ['clu431', 'base_cnn', 0.6186690491603315, 0.6581632653061225],
        ['clu3', 'base_cnn', 0.6686813186813186, 0.5714285714285714],
        ['clu1', 'base_cnn', 0.641804504251939, 0.6173469387755102],

        ['clu4', 'cnn', 0.9075065179958857, 0.6581632653061225],
        ['clu431', 'cnn', 0.6418139209195343, 0.7346938775510203],
        ['clu3', 'cnn', 0.6452047413793103, 0.5408163265306123],
        ['clu1', 'cnn', 0.6508022686394104, 0.7142857142857143],

        ['clu4', 'rnn', 0.8232919272012025, 0.6122448979591837],
        ['clu431', 'rnn', 0.6484552665751815, 0.7193877551020409],
        ['clu3', 'rnn', 0.6410290948275862, 0.6377551020408163],
        ['clu1', 'rnn', 0.6498387859405184, 0.7040816326530612],

        ['clu4', 'transformer', 0.9120521172638436, 0.5816326530612245],
        ['clu431', 'transformer', 0.4660534776284099, 0.5663265306122449],
        ['clu3', 'transformer', 0.4171605603448276, 0.29591836734693877],
        ['clu1', 'transformer', 0.5817958179581796, 0.7397959183673469],
    ]
    
    if plot_acc_comparisons:
        plot_data(destination, data, 'clu_pooled_accuracies.png', color_map, plot_location='upper left')
    if plot_amounts_vs_acc:
        plot_amounts_against_data_size(destination, data, 'pooled_test_accuracies_against_data_size.png', color_map, 3, 'Test')
        plot_amounts_against_data_size(destination, data, 'pooled_valid_accuracies_against_data_size.png', color_map, 2, 'Valid')
        
    
    
    
if __name__=='__main__':
    main()