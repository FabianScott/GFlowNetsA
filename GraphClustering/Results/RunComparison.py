from GraphClustering.Core.Core import compare_results

if __name__ == '__main__':
    N = 6
    compare_results(filename=f'Comparison{N}.txt', min_N=N, max_N=N)
