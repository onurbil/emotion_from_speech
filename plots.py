import matplotlib.pyplot as plt
import numpy as np

labels = ['RNN', 'Bi-RNN', 'LSTM', 'Bi-LSTM', 'GRU', 'Bi-GRU']
one_layer_means = [0.187777777777777, 0.230555555555555, 0.87, 0.902777777777777, 0.925555555555555, 0.751666666666667]
one_layer_maxs = [0.213333333333333, 0.33, 0.973333333333333, 0.988333333333333, 0.993333333333333, 0.99]
two_layers_means = [0.243888888888889, 0.236111111111111, 0.837777777777778, 0.566666666666666, 0.7, 0.936666666666666]
two_layers_maxs = [0.303333333333333, 0.273333333333333, 0.995, 0.818333333333333, 0.993333333333333, 0.99]
three_layers_means = [0.184999999999999, 0.135555555555555, 0.285, 0.386666666666666, 0.806666666666666, 0.595555555555555]
three_layers_maxs = [0.278333333333333, 0.138333333333333, 0.531666666666666, 0.82, 0.976666666666666, 0.836666666666666]
hidden_512_means = [0.26, 0.213333333333333, 0.800555555555555, 0.986666666666666, 0.975, 0.931111111111111]
hidden_512_maxs = [0.311666666666666, 0.261666666666666, 0.983333333333333, 0.991666666666666, 0.98, 0.991666666666666]
hidden_1024_means = [0.177222222222222, 0.220555555555555, 0.839444444444444, 0.978888888888888, 0.845, 0.895]
hidden_1024_maxs = [0.218333333333333, 0.23, 0.99, 0.981666666666666, 0.968333333333333, 0.973333333333333]


def plot_accuracies(x_labels, values_lists, value_labels, title, file_name=None):
    x = np.arange(len(labels))

    fig, ax = plt.subplots()

    if len(values_lists) == 2:
        width = 0.3
        rects1 = ax.bar(x - width, values_lists[0], width, label=value_labels[0])
        rects3 = ax.bar(x + width, values_lists[1], width, label=value_labels[1])
    elif len(values_lists) == 3:
        width = 0.20
        rects1 = ax.bar(x - width, values_lists[0], width, label=value_labels[0])
        rects2 = ax.bar(x, values_lists[1], width, label=value_labels[1])
        rects3 = ax.bar(x + width, values_lists[2], width, label=value_labels[2])

    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    fig.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    plt.show()


layers_means = [one_layer_means, two_layers_means, three_layers_means]
layers_maxs = [one_layer_maxs, three_layers_maxs, two_layers_maxs]
size_means = [one_layer_means, hidden_512_means, hidden_1024_means]
size_maxs = [one_layer_maxs, hidden_512_maxs, hidden_1024_maxs]

layers_labels = ['1 layer', '2 layers', '3 layers']
layers_means_title = 'Average model accuracies by number of layers'
layers_maxs_title = 'Maximum model accuracies by number of layers'
sizes_labels = ['hidden size: 256', 'hidden size: 512', 'hidden size: 1024']
size_means_title = 'Average model accuracies by hidden size'
size_maxs_title = 'Maximum model accuracies by hidden'

plot_accuracies(labels, layers_means, layers_labels, layers_means_title, file_name='plots/mean_accuracy_layer.png')
plot_accuracies(labels, layers_maxs, layers_labels, layers_maxs_title, file_name='plots/max_accuracy_layer.png')

plot_accuracies(labels, size_means, sizes_labels, size_means_title, file_name='plots/mean_accuracy_hidden_size.png')
plot_accuracies(labels, size_maxs, sizes_labels, size_maxs_title, file_name='plots/max_accuracy_hidden_size.png')
