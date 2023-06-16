import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GraphClustering import *


def errorPlot(values, labels, alpha=1., prefix='', transfer=''):

    for row, label in zip(values, labels):
        plt.plot(row[:-1], label=label, alpha=alpha)
    plt.legend()
    plt.title(f'GFlowNet Error for randomly generated graph of '
              f'{labels[0] if len(labels) == 1 else str(labels[0]) + "-" + str(labels[-1])} nodes over time ' + transfer)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.savefig(f'Plots/{prefix}ErrorPlot{labels[0] if len(labels) == 1 else str(labels[0]) + "_" + str(labels[-1])}{transfer}.png')
    plt.show()


if __name__ == '__main__':

    alpha = 1.
    max_epochs = '99'
    prefix = f'{max_epochs}_'
    df = pd.read_csv(f'Data/New_Final_Comparison_test_o_{max_epochs}_1_2_5.txt', sep=',')
    df2 = pd.read_csv(f'Data/New_Final_Comparison_test_o_{max_epochs}_1_6_6.txt', sep=',')
    df3 = pd.read_csv(f'Data/New_Final_Comparison_test_o_{max_epochs}_1_2_5_TEST.csv', sep=',')
    df4 = pd.read_csv(f'Data/New_Final_Comparison_test_o_{max_epochs}_1_6_6_TEST.csv', sep=',')

    errorPlot(df.values[:-1], range(2, 5), prefix=prefix)
    errorPlot([df.values[-1]], labels=['5'], prefix=prefix)
    errorPlot([df2.values[-1]], labels=['6'], prefix=prefix)

    errorPlot(df3.values[:-1], range(2, 5), prefix=prefix, transfer='t')
    errorPlot([df3.values[-1][1:]], labels=['5'], prefix=prefix, transfer='t')
    errorPlot([df4.values[-1][1:]], labels=['6'], prefix=prefix, transfer='t')
