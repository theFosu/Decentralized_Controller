import warnings

import graphviz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import os


def plot_stats(df, ylog=False, view=False, filename='Graphs/clipped_fitness_comparison'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    mpl.use('TkAgg')

    generations = np.arange(1, 152, 5)

    df[0]['Generation'] = generations
    df[1]['Generation'] = generations
    df[2]['Generation'] = generations

    plt.plot(df[0]['Generation'], df[0]['median_eval'], color='blue', label='DecLoco')
    plt.fill_between(df[0]['Generation'], df[0]['best_eval'], df[0]['worst_eval'], color='lightblue', alpha=0.8)

    plt.plot(df[1]['Generation'], df[1]['median_eval'], color='red', label='SCN-0')
    plt.fill_between(df[1]['Generation'], df[1]['best_eval'], df[1]['worst_eval'], color='lightcoral', alpha=0.4)

    plt.plot(df[2]['Generation'], df[2]['median_eval'], color='green', label='MLP-4')
    plt.fill_between(df[2]['Generation'], df[2]['best_eval'], df[2]['worst_eval'], color='lightgreen', alpha=0.6)

    # Adding a legend
    plt.legend()

    # plt.axvline(x=threshold, color='black', linestyle='--', label="phase threshold")

    plt.title("Fitness progression")
    plt.xlabel("Generations")
    plt.ylabel("logFitness")
    #plt.grid()
    plt.legend(loc="center")
    plt.xlim(1, 150)
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


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
    main_df = get_main()

    nonei_df = pd.read_pickle('less-Checkpoints/lessDataframe.pickle')

    ff_df = pd.read_pickle('ff-Checkpoints/ffDataframe.pickle')
    ff_df = ff_df.iloc[:31, :]

    keys = ['best_eval', 'worst_eval', 'median_eval']
    print(ff_df['best_eval'])

    df = [main_df[keys], nonei_df[keys], ff_df[keys]]
    plot_stats(df, view=True, ylog=True)


if __name__ == "__main__":
    main()
