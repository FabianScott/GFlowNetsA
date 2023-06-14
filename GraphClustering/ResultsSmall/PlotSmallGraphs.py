import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('Data/New_Final_Comparison_test_o_101_1_2_5.txt', sep=',')
    df2 = pd.read_csv('Data/New_Final_Comparison_test_o_101_1_6_6.txt', sep=',')

    for i, row in enumerate(df.values[:-1]):
        plt.plot(row[:-1], label=f'{i+2}')
    plt.legend()
    plt.title('Error for randomly generated graph of 2-4 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/ErrorPlot2_4.png')
    plt.show()

    plt.plot(df.values[-1], label=f'5')
    plt.legend()
    plt.title('Error for randomly generated graph of 5 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/ErrorPlot5.png')
    plt.show()

    plt.plot(df2.values[-1], label=f'5')
    plt.legend()
    plt.title('Error for randomly generated graph of 6 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/ErrorPlot6.png')
    plt.show()
