import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    n_samples = 1000
    epoch_interval = 100
    min_epochs = 0
    max_epochs = 100

    df_list = []

    for i in range(4):
        filename = f'Data/KarateResults_{min_epochs + i * epoch_interval}_{max_epochs + i * epoch_interval}_{n_samples}_o.csv'
        df = pd.read_csv(filename, header=None)
        df_list.append(df[:10])

    df_final = pd.concat(tuple(df_list))

    plt.plot(range(0, 400, 10), df_final.values[:, 1], label='Error')
    plt.title(f'Error over time on Karate Graph with {n_samples} samples')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Error')
    plt.ylim(0, 1000)
    plt.legend()
    plt.show()

