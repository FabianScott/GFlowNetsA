import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    alpha = .8
    df = pd.read_csv('Data/New_Final_Comparison_test_o_101_1_2_5.txt', sep=',')
    df2 = pd.read_csv('Data/New_Final_Comparison_test_o_101_1_6_6.txt', sep=',')
    df3 = pd.read_csv('Data/New_Final_Comparison_test_o_101_1_2_5_TEST.csv', sep=',')
    df4 = pd.read_csv('Data/New_Final_Comparison_test_o_101_1_6_6_TEST.csv', sep=',')

    for i, row in enumerate(df.values[:-1]):
        plt.plot(row[:-1], label=f'{i+2}', alpha=alpha)
    plt.legend()
    plt.title('GFlowNet Error for randomly generated graph of 2-4 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/ErrorPlot2_4.png')
    plt.show()

    plt.plot(df.values[-1], label=f'5', alpha=alpha)
    plt.legend()
    plt.title('GFlowNet Error for randomly generated graph of 5 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/ErrorPlot5.png')
    plt.show()

    plt.plot(df2.values[-1], label=f'6', alpha=alpha)
    plt.legend()
    plt.title('GFlowNet Error for randomly generated graph of 6 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/ErrorPlot6.png')
    plt.show()

    for i, row in enumerate(df.values[:-1]):
        plt.plot(row[:-1], label=f'{i+2}', alpha=alpha)
    plt.legend()
    plt.title('GFlowNet Error for randomly generated graph of 2-4 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/TransferErrorPlot2_4.png')
    plt.show()

    plt.plot(df.values[-1], label=f'5', alpha=alpha)
    plt.legend()
    plt.title('GFlowNet Error for randomly generated graph of 5 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/TransferErrorPlot5.png')
    plt.show()

    plt.plot(df2.values[-1], label=f'6', alpha=alpha)
    plt.legend()
    plt.title('GFlowNet Error for randomly generated graph of 6 nodes over time')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig('Plots/TransferErrorPlot6.png')
    plt.show()

