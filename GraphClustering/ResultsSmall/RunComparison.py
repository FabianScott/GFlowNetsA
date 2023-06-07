try:
    from Core import compare_results_small_graphs
except ModuleNotFoundError:
    from GraphClustering.Core.Core import compare_results_small_graphs

if __name__ == '__main__':
    N = 6
    max_epochs = 300
    compare_results_small_graphs(filename=f'Comparison_test_{max_epochs}_{N}.txt', min_N=N, max_N=N, max_epochs=max_epochs, use_new_graph_for_test=True)
