import warnings

import graphviz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os


def plot_stats(df, ylog=False, view=False, filename='Graphs/clipped_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    mpl.use('TkAgg')

    generations = np.arange(1, 152, 5)
    print(generations)
    df['Generation'] = generations

    plt.plot(df['Generation'], df['median_eval'], color='blue', label='Population Center Fitness')

    # Plotting the shaded area for best and worst fitness
    plt.fill_between(df['Generation'], df['best_eval'], df['worst_eval'], color='lightblue', alpha=0.4, label='Total Fitness Range')

    # Adding a legend
    plt.legend()

    # plt.axvline(x=threshold, color='black', linestyle='--', label="phase threshold")

    plt.title("Fitness progression")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    #plt.grid()
    plt.legend(loc="center")
    plt.xlim(1, 150)
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename=filename, view=view)

    return dot


def get_main():
    folder_path = 'new-Checkpoints/'

    # Get the list of pickle files in the folder
    pickle_files = [file for file in os.listdir(folder_path)]

    # Sort the list of pickle files
    pickle_files = sorted(pickle_files)

    full_dict = {'mean_eval': [-168.22711905855635], 'pop_best_eval': [-111.63850514171196],
                 'median_eval': [-166.00387061608035], 'best_eval': [-111.63850514171196],
                 'worst_eval': [-259.555534683086]}

    # Iterate through each file in the folder
    for i, file in enumerate(pickle_files):

        file_path = os.path.join(folder_path, file)  # Get the full file path

        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
            for key in full_dict.keys():
                full_dict[key].append(data_dict[key])
        if i >= 29:
            break

    return pd.DataFrame.from_dict(full_dict)


def main():
    df = get_main()
    print(df['best_eval'])

    keys = ['best_eval', 'worst_eval', 'median_eval']
    # print(df.columns)
    plot_stats(df[keys], view=True)


if __name__ == "__main__":
    main()
