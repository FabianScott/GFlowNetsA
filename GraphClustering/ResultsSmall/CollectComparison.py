from GraphClustering.Core.Core import print_Latex_table
import pandas as pd


if __name__ == '__main__':
    df25 = pd.read_csv('Comparison_300_25.txt.txt', sep=',', index_col=0)
    df5 = pd.read_csv('Comparison5.txt', sep=',', index_col=0)
    df6 = pd.read_csv('Comparison6.txt', sep=',', index_col=0)
    df26 = pd.concat((df25, df5, df6))
    df26.to_csv('ResultTable2_6.csv', sep=',')

    print_Latex_table(df26.values, significantFigures=3, headerRow=range(0, 101, 10), indexColumn=[str(el) + ' & ' for el in range(2,7)])





