import matplotlib.pyplot as plt
import numpy as np

labels = ['RNN', 'Bi-RNN', 'LSTM', 'Bi-LSTM', 'GRU', 'Bi-GRU']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
hatches = ['', '//////', '....']


def get_clean_order1_results():
    one_layer_means = [0.187777777777777, 0.230555555555555, 0.87, 0.902777777777777, 0.925555555555555,
                       0.751666666666667]
    one_layer_maxs = [0.213333333333333, 0.33, 0.973333333333333, 0.988333333333333, 0.993333333333333, 0.99]
    two_layers_means = [0.243888888888889, 0.236111111111111, 0.837777777777778, 0.566666666666666, 0.7,
                        0.936666666666666]
    two_layers_maxs = [0.303333333333333, 0.273333333333333, 0.995, 0.818333333333333, 0.993333333333333, 0.99]
    three_layers_means = [0.184999999999999, 0.135555555555555, 0.285, 0.386666666666666, 0.806666666666666,
                          0.595555555555555]
    three_layers_maxs = [0.278333333333333, 0.138333333333333, 0.531666666666666, 0.82, 0.976666666666666,
                         0.836666666666666]
    hidden_512_means = [0.26, 0.213333333333333, 0.800555555555555, 0.986666666666666, 0.975, 0.931111111111111]
    hidden_512_maxs = [0.311666666666666, 0.261666666666666, 0.983333333333333, 0.991666666666666, 0.98,
                       0.991666666666666]
    hidden_1024_means = [0.177222222222222, 0.220555555555555, 0.839444444444444, 0.978888888888888, 0.845, 0.895]
    hidden_1024_maxs = [0.218333333333333, 0.23, 0.99, 0.981666666666666, 0.968333333333333, 0.973333333333333]
    return (one_layer_means,
            one_layer_maxs,
            two_layers_means,
            two_layers_maxs,
            three_layers_means,
            three_layers_maxs,
            hidden_512_means,
            hidden_512_maxs,
            hidden_1024_means,
            hidden_1024_maxs)


def get_clean_order2_results():
    one_layer_means = [0.264444444444444, 0.167777777777777, 0.937222222222222, 0.740555555555555, 0.982777777777777,
                       0.764444444444444]
    one_layer_maxs = [0.291666666666666, 0.248333333333333, 0.98, 0.843333333333333, 0.99, 0.99]
    two_layers_means = [0.226111111111111, 0.271666666666667, 0.642222222222222, 0.815555555555555, 0.638333333333333,
                        0.841111111111111]
    two_layers_maxs = [0.256666666666666, 0.28, 0.96, 0.976666666666666, 0.968333333333333, 0.995]
    three_layers_means = [0.184999999999999, 0.135555555555555, 0.285, 0.386666666666666, 0.806666666666666,
                          0.595555555555555]
    three_layers_maxs = [0.27, 0.243333333333333, 0.438333333333333, 0.831666666666666, 0.98, 0.948333333333333]
    hidden_512_means = [0.26, 0.213333333333333, 0.800555555555555, 0.986666666666666, 0.975, 0.931111111111111]
    hidden_512_maxs = [0.261666666666666, 0.2, 0.971666666666666, 0.99, 0.991666666666666, 0.986666666666666]
    hidden_1024_means = [0.177222222222222, 0.220555555555555, 0.839444444444444, 0.978888888888888, 0.845, 0.895]
    hidden_1024_maxs = [0.241666666666666, 0.208333333333333, 0.993333333333333, 0.993333333333333, 0.991666666666666,
                        0.991666666666666]
    return (one_layer_means,
            one_layer_maxs,
            two_layers_means,
            two_layers_maxs,
            three_layers_means,
            three_layers_maxs,
            hidden_512_means,
            hidden_512_maxs,
            hidden_1024_means,
            hidden_1024_maxs)


def get_augmented_order1_results():
    one_layer_aug_means = [0.231666666666667, 0.241666666666667, 0.91, 0.915, 0.815, 0.845]
    one_layer_aug_maxs = [0.325, 0.34, 0.92, 0.92, 0.91, 0.955]
    one_layer_clean_means = [0.265, 0.269444444444444, 0.961666666666666, 0.947777777777778, 0.866666666666666,
                             0.885555555555555]
    one_layer_clean_maxs = [0.355, 0.333333333333333, 0.973333333333333, 0.965, 0.955, 0.971666666666666]
    two_layers_aug_means = [0.183333333333333, 0.206666666666667, 0.505, 0.426666666666667, 0.718333333333333,
                            0.721666666666667]
    two_layers_aug_maxs = [0.22, 0.255, 0.725, 0.72, 0.855, 0.915]
    two_layers_clean_means = [0.186111111111111, 0.23, 0.505555555555555, 0.463888888888888, 0.755, 0.745555555555555]
    two_layers_clean_maxs = [0.221666666666666, 0.243333333333333, 0.738333333333333, 0.771666666666666,
                             0.913333333333333, 0.925]
    three_layers_aug_means = [0.223333333333333, 0.186666666666667, 0.255, 0.198333333333333, 0.53, 0.516666666666667]
    three_layers_aug_maxs = [0.34, 0.23, 0.42, 0.365, 0.73, 0.68]
    three_layers_clean_means = [0.227222222222222, 0.188333333333333, 0.255, 0.218333333333333, 0.573888888888888,
                                0.514444444444444]
    three_layers_clean_maxs = [0.313333333333333, 0.22, 0.456666666666666, 0.421666666666666, 0.796666666666666, 0.66]
    hidden_512_aug_means = [0.19, 0.241666666666667, 0.915, 0.901666666666667, 0.833333333333333, 0.613333333333333]
    hidden_512_aug_maxs = [0.2, 0.3, 0.935, 0.925, 0.855, 0.975]
    hidden_512_clean_means = [0.220555555555555, 0.233888888888889, 0.957222222222222, 0.953333333333333,
                              0.897777777777778, 0.625]
    hidden_512_clean_maxs = [0.241666666666666, 0.291666666666666, 0.961666666666666, 0.97, 0.92, 0.97]
    hidden_1024_aug_means = [0.233333333333333, 0.146666666666667, 0.866666666666667, 0.861666666666667, 0.465,
                             0.748333333333333]
    hidden_1024_aug_maxs = [0.3, 0.165, 0.965, 0.885, 0.93, 0.925]
    hidden_1024_clean_means = [0.217222222222222, 0.156666666666666, 0.89, 0.938888888888889, 0.470555555555555,
                               0.781666666666666]
    hidden_1024_clean_maxs = [0.27, 0.175, 0.98, 0.951666666666666, 0.945, 0.956666666666666]
    return (one_layer_aug_means, one_layer_aug_maxs, one_layer_clean_means, one_layer_clean_maxs,
            two_layers_aug_means, two_layers_aug_maxs, two_layers_clean_means, two_layers_clean_maxs,
            three_layers_aug_means, three_layers_aug_maxs, three_layers_clean_means, three_layers_clean_maxs,
            hidden_512_aug_means, hidden_512_aug_maxs, hidden_512_clean_means, hidden_512_clean_maxs,
            hidden_1024_aug_means, hidden_1024_aug_maxs, hidden_1024_clean_means, hidden_1024_clean_maxs)


def get_augmented_order2_results():
    one_layer_aug_means = [0.241111111111111, 0.214444444444444, 0.905, 0.856111111111111, 0.876666666666666,
                           0.906111111111111]
    one_layer_aug_maxs = [0.248333333333333, 0.261666666666666, 0.935, 0.925, 0.916666666666666, 0.93]
    one_layer_clean_means = [0.239444444444444, 0.218333333333333, 0.934444444444444, 0.898888888888889,
                             0.919444444444444, 0.927222222222222]
    one_layer_clean_maxs = [0.243333333333333, 0.27, 0.963333333333333, 0.95, 0.953333333333333, 0.946666666666666]
    two_layers_aug_means = [0.251666666666666, 0.256111111111111, 0.67, 0.578888888888888, 0.836666666666666,
                            0.732222222222222]
    two_layers_aug_maxs = [0.253333333333333, 0.28, 0.85, 0.661666666666666, 0.9, 0.891666666666666]
    two_layers_clean_means = [0.248888888888889, 0.258888888888889, 0.733888888888889, 0.609444444444444,
                              0.857222222222222, 0.767222222222222]
    two_layers_clean_maxs = [0.26, 0.286666666666666, 0.875, 0.7, 0.938333333333333, 0.905]
    three_layers_aug_means = [0.238333333333333, 0.241111111111111, 0.126111111111111, 0.242777777777777, 0.43,
                              0.187777777777777]
    three_layers_aug_maxs = [0.258333333333333, 0.255, 0.135, 0.448333333333333, 0.915, 0.298333333333333]
    three_layers_clean_means = [0.243333333333333, 0.233333333333333, 0.126111111111111, 0.242222222222222,
                                0.440555555555555, 0.191111111111111]
    three_layers_clean_maxs = [0.256666666666666, 0.251666666666666, 0.135, 0.441666666666666, 0.933333333333333,
                               0.308333333333333]
    hidden_512_aug_means = [0.188888888888889, 0.189444444444444, 0.833888888888889, 0.811666666666666,
                            0.863333333333333, 0.880555555555555]
    hidden_512_aug_maxs = [0.255, 0.253333333333333, 0.885, 0.903333333333333, 0.886666666666666, 0.955]
    hidden_512_clean_means = [0.19, 0.19, 0.879999999999999, 0.862777777777777, 0.889444444444444, 0.927777777777778]
    hidden_512_clean_maxs = [0.255, 0.255, 0.936666666666666, 0.955, 0.931666666666666, 0.963333333333333]
    hidden_1024_aug_means = [0.148888888888889, 0.155555555555555, 0.857222222222222, 0.867222222222222, 0.58,
                             0.252777777777777]
    hidden_1024_aug_maxs = [0.16, 0.205, 0.925, 0.906666666666666, 0.893333333333333, 0.38]
    hidden_1024_clean_means = [0.154444444444444, 0.155555555555555, 0.899444444444444, 0.911111111111111,
                               0.617222222222222, 0.264999999999999]
    hidden_1024_clean_maxs = [0.165, 0.206666666666666, 0.96, 0.958333333333333, 0.935, 0.416666666666666]
    return (one_layer_aug_means, one_layer_aug_maxs, one_layer_clean_means, one_layer_clean_maxs,
            two_layers_aug_means, two_layers_aug_maxs, two_layers_clean_means, two_layers_clean_maxs,
            three_layers_aug_means, three_layers_aug_maxs, three_layers_clean_means, three_layers_clean_maxs,
            hidden_512_aug_means, hidden_512_aug_maxs, hidden_512_clean_means, hidden_512_clean_maxs,
            hidden_1024_aug_means, hidden_1024_aug_maxs, hidden_1024_clean_means, hidden_1024_clean_maxs)


def plot_accuracies(x_labels, values_lists, value_labels, hatch_every=None, title=None, file_name=None):
    x = np.arange(len(labels))

    num_values = len(values_lists)
    fig_width = 6 if num_values <= 3 else 8
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    if num_values == 2:
        width = 0.3
        rects1 = ax.bar(x - width, values_lists[0], width, label=value_labels[0])
        rects3 = ax.bar(x + width, values_lists[1], width, label=value_labels[1])
    elif num_values == 3:
        width = 0.20
        rects1 = ax.bar(x - width, values_lists[0], width, label=value_labels[0], color=colors[0])
        rects2 = ax.bar(x, values_lists[1], width, label=value_labels[1], color=colors[1])
        rects3 = ax.bar(x + width, values_lists[2], width, label=value_labels[2], color=colors[2])
    else:
        width = 0.6 / num_values
        base_position = -width * (num_values - 1) / 2
        for i, values in enumerate(values_lists):
            position = base_position + i * width
            color = colors[i % len(colors)]
            edge_color = color
            hatch = None
            if hatch_every is not None:
                color = colors[int(i / hatch_every) % len(colors)]
                edge_color = color
                hatch = hatches[i % hatch_every]
                if hatch != '':
                    color = 'white'
            rects = ax.bar(x + position, values, width, label=value_labels[i], color=color, hatch=hatch,
                           edgecolor=edge_color)

    if title is not None:
        ax.set_title(title)

    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    fig.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    plt.show()


def create_plots_single_dataset(results, base_name):
    (one_layer_means, one_layer_maxs,
     two_layers_means, two_layers_maxs,
     three_layers_means, three_layers_maxs,
     hidden_512_means, hidden_512_maxs,
     hidden_1024_means, hidden_1024_maxs) = results

    layers_means = [one_layer_means, two_layers_means, three_layers_means]
    layers_maxs = [one_layer_maxs, two_layers_maxs, three_layers_maxs]
    size_means = [one_layer_means, hidden_512_means, hidden_1024_means]
    size_maxs = [one_layer_maxs, hidden_512_maxs, hidden_1024_maxs]

    layers_labels = ['1 layer', '2 layers', '3 layers']
    sizes_labels = ['hidden size: 256', 'hidden size: 512', 'hidden size: 1024']

    plot_accuracies(labels, layers_means, layers_labels,
                    file_name=f'plots/{base_name}_mean_accuracy_layer.png')
    plot_accuracies(labels, layers_maxs, layers_labels,
                    file_name=f'plots/{base_name}_max_accuracy_layer.png')

    plot_accuracies(labels, size_means, sizes_labels,
                    file_name=f'plots/{base_name}_mean_accuracy_hidden_size.png')
    plot_accuracies(labels, size_maxs, sizes_labels,
                    file_name=f'plots/{base_name}_max_accuracy_hidden_size.png')


def create_plots_two_orders(results1, results2, results1_name, results2_name, base_name):
    (one_layer_means1, one_layer_maxs1,
     two_layers_means1, two_layers_maxs1,
     three_layers_means1, three_layers_maxs1,
     hidden_512_means1, hidden_512_maxs1,
     hidden_1024_means1, hidden_1024_maxs1) = results1

    (one_layer_means2, one_layer_maxs2,
     two_layers_means2, two_layers_maxs2,
     three_layers_means2, three_layers_maxs2,
     hidden_512_means2, hidden_512_maxs2,
     hidden_1024_means2, hidden_1024_maxs2) = results2

    layers_means = [one_layer_means1, one_layer_means2, two_layers_means1, two_layers_means2, three_layers_means1,
                    three_layers_maxs2]
    layers_maxs = [one_layer_maxs1, one_layer_maxs2, two_layers_maxs1, two_layers_maxs2, three_layers_maxs1,
                   three_layers_maxs2]
    size_means = [one_layer_means1, one_layer_means2, hidden_512_means1, hidden_512_means2, hidden_1024_means1,
                  hidden_1024_means2]
    size_maxs = [one_layer_maxs1, one_layer_maxs2, hidden_512_maxs1, hidden_512_maxs2, hidden_1024_maxs1,
                 hidden_1024_maxs2]

    layers_labels = [f'1 layer, {results1_name}', f'1 layer, {results2_name}',
                     f'2 layers, {results1_name}', f'2 layers, {results2_name}',
                     f'3 layers, {results1_name}', f'3 layers, {results2_name}']
    sizes_labels = [f'hidden size: 256, {results1_name}', f'hidden size: 256, {results2_name}',
                    f'hidden size: 512, {results1_name}', f'hidden size: 512, {results2_name}',
                    f'hidden size: 1024, {results1_name}', f'hidden size: 1024, {results2_name}']

    plot_accuracies(labels, layers_means, layers_labels, hatch_every=2,
                    file_name=f'plots/{base_name}_mean_accuracy_layer.png')
    plot_accuracies(labels, layers_maxs, layers_labels, hatch_every=2,
                    file_name=f'plots/{base_name}_max_accuracy_layer.png')

    plot_accuracies(labels, size_means, sizes_labels, hatch_every=2,
                    file_name=f'plots/{base_name}_mean_accuracy_hidden_size.png')
    plot_accuracies(labels, size_maxs, sizes_labels, hatch_every=2,
                    file_name=f'plots/{base_name}_max_accuracy_hidden_size.png')


def create_plots_two_datasets_two_orders(results1, results2, results1_name, results2_name, base_name1, base_name2):
    results_aug1 = []
    results_aug2 = []
    results_clean1 = []
    results_clean2 = []
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        if i % 4 == 0 or i % 4 == 1:
            results_aug1.append(r1)
            results_aug2.append(r2)
        else:
            results_clean1.append(r1)
            results_clean2.append(r2)

    create_plots_two_orders(results_aug1, results_aug2, results1_name, results2_name, base_name1)
    create_plots_two_orders(results_clean1, results_clean2, results1_name, results2_name, base_name2)


# create_plots_single_dataset(get_clean_order1_results(), 'clean_order1')
# create_plots_single_dataset(get_clean_order2_results(), 'clean_order2')

create_plots_two_orders(get_clean_order1_results(), get_clean_order2_results(), 'order 1', 'order 1+2', 'clean_both_orders')
create_plots_two_datasets_two_orders(get_augmented_order1_results(), get_augmented_order2_results(),
                                     'order 1', 'order 1+2',
                                     'train-aug_test-aug', 'train-aug_test-clean')
