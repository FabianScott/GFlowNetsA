import pandas as pd

try:
    from Core import compare_results_small_graphs, print_Latex_table
except ModuleNotFoundError:
    from GraphClustering import compare_results_small_graphs, print_Latex_table

if __name__ == '__main__':
    min_N = 6
    max_N = 6
    max_epochs = 99
    epoch_interval = 1
    n_samples = 1000
    use_node_order = True

    node_order_string = 'o_' if use_node_order else ''
    fname = f'Data/New_Final_Comparison_test_{node_order_string}{max_epochs}_{epoch_interval}_{min_N}_{max_N}.txt'
    networks = compare_results_small_graphs(filename=fname,
                                            min_N=min_N,
                                            max_N=max_N,
                                            run_test=True,
                                            plot_last=True,
                                            n_samples=n_samples,
                                            n_samples_distribution=n_samples,
                                            max_epochs=max_epochs,
                                            epoch_interval=epoch_interval,
                                            use_fixed_node_order=use_node_order)

