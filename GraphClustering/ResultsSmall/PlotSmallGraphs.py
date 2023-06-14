import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('Data/Final_Comparison_test_o_100_1_2_5.txt', sep=',')

    for i, row in enumerate(df.values[:-1]):
        plt.plot(row[:-1], label=f'{i+2}')
    plt.legend()
    plt.title('Error for randomly generated graph of 2-4 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('ErrorPlot2_4.png')
    plt.show()

