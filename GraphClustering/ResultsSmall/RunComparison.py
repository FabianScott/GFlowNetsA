import pandas as pd

try:
    from Core import compare_results_small_graphs, print_Latex_table
except ModuleNotFoundError:
    from GraphClustering.Core.Core import compare_results_small_graphs, print_Latex_table

if __name__ == '__main__':
    min_N = 2
    max_N = 5
    max_epochs = 100
    epoch_interval = 1
    n_samples = 100
    use_node_order = True

    node_order_string = 'o_' if use_node_order else ''
    fname = f'Data/Comparison_test_{node_order_string}{max_epochs}_{epoch_interval}_{min_N}_{max_N}.txt'
    networks = compare_results_small_graphs(filename=fname,
                                            min_N=min_N,
                                            max_N=max_N,
                                            run_test=True,
                                            plot_last=True,
                                            n_samples=n_samples,
                                            max_epochs=max_epochs,
                                            epoch_interval=epoch_interval,
                                            use_fixed_node_order=use_node_order)
    df26 = pd.read_csv(fname, sep=',', index_col=0)
    print_Latex_table(df26.values, significantFigures=3,
                      headerRow=range(0, max_epochs + epoch_interval, epoch_interval),
                      indexColumn=[str(el) + ' & ' for el in range(min_N, max_N + 1)])
