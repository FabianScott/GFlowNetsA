from GraphClustering.Core.Core import compare_results_small_graphs
import pandas as pd

if __name__ == '__main__':
    df24 = pd.read_csv('Comparison24.txt', sep=',', index_col=0)
    df5 = pd.read_csv('Comparison5.txt', sep=',', index_col=0)
    df6 = pd.read_csv('Comparison6.txt', sep=',', index_col=0)
    df26 = pd.concat((df24, df5, df6))
    df26.to_csv('ResultTable2_6.csv', sep=',')



