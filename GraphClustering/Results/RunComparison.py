from GraphClustering.Core.Core import compare_results_small_graphs

if __name__ == '__main__':
    N = 6
    compare_results_small_graphs(filename=f'Comparison{N}.txt', min_N=N, max_N=N)
