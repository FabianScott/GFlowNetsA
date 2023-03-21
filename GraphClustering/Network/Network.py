import numpy as np
import torch
import torch.nn as nn
import itertools


class GraphNet:
    def __init__(self,
                 env=None,
                 n_layers=2,
                 n_hidden=32,
                 gamma=0.5,
                 epochs=100,
                 lr=0.005,
                 # decay_steps=10000,
                 # decay_rate=0.8,
                 n_samples=1000,
                 batch_size=10,
                 n_clusters=4,
                 n_nodes=100,
                 using_cuda=False,
                 using_backward_model=False
                 ):
        self.env = env
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gamma = gamma  # weighting of random sampling if applied
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.model_forward = MLP(output_size=int(n_nodes * 2),
                                 n_nodes=n_nodes,
                                 n_hidden=n_hidden,
                                 n_clusters=n_clusters,
                                 n_layers=n_layers)

        self.model_backward = MLP(output_size=int(n_nodes),
                                  n_nodes=n_nodes,
                                  n_hidden=n_hidden,
                                  n_clusters=n_clusters,
                                  n_layers=n_layers) if using_backward_model\
            else SimpleBackwardModel()
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=0)
        self.z0 = nn.Parameter(torch.tensor([.0]))
        self.optimizer = torch.optim.Adam(itertools.chain(self.model_forward.parameters(), (self.z0,)), lr=self.lr)
        self.using_cuda = using_cuda
        self.using_backward_model = using_backward_model

    def create_model(self):
        # Define the layers of the neural network
        # layers = []
        # Assuming the features extracted for each cluster has size 1

        # Create an instance of the neural network and return it
        # net = nn.Module()
        return MLP(n_hidden=self.n_hidden, n_clusters=self.n_clusters, n_layers=self.n_layers, output_size=1)

    def train(self, X, Y, complete_graphs=None, epochs=100, batch_size=None):
        # X: an iterable/index-able of final cluster assignments
        # Y: an iterable/index-able of IRM values for each X
        if batch_size is None:
            batch_size = self.batch_size
        if complete_graphs is None:
            print(f'Missing iterable indicating which graphs can be evaluated using IRM!')
            raise NotImplementedError

        permutation = torch.randperm(X.size()[0])
        for epoch in range(epochs):
            for i in range(0, X.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X[indices], Y[indices]

                outputs = self.model_forward.forward(batch_x)
                loss = self.mse_loss(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

    def trajectory_balance_loss(self, final_states):
        # final_states is a list of vectors representing states with at minimum one node missing
        for state in final_states:
            # The probabilities are computed while creating the trajectories
            trajectory, backward_probs, forward_probs = self.sample_trajectory_backwards(state)


    def sample_trajectory_backwards(self, state):
        # state is the (k ** 2) * 2 long vector of both matrices
        # reshape stuff to make it easier to work with

        current_adjacency, current_clustering, size = self.get_matrices_from_state(state, return_size=True)
        zeros = torch.zeros(size[0])
        zeros_ = torch.zeros((size[0], 1))

        trajectory, backward_probs_mat, forward_probs_mat = [torch.zeros(size)] * 3
        prob_sum_forward, prob_sum_backward = 0, 0
        probs_backward = self.model_backward.forward(current_clustering)

        for i in range(size[0]):
            # put the backward probabilities in at the start of the loop to avoid the last one
            backward_probs_mat[i] = probs_backward

            index_chosen = torch.multinomial(probs_backward, 1)
            # remove the chosen node from the clustering
            current_clustering[index_chosen] = zeros
            current_clustering[:, index_chosen] = zeros_
            
            prob_sum_backward += probs_backward[index_chosen]
            probs_backward = self.model_backward.forward(current_clustering)

            # put the forward probabilities in at the end to skip the first iteration but include the last
            current_state = self.get_state_from_matrices(current_adjacency, current_clustering)
            probs_forward = self.model_forward.forward(current_state)
            forward_probs_mat[i] = probs_forward
            prob_sum_forward += probs_forward[index_chosen]
            trajectory[i][index_chosen] = 1
        # trajectory[i + 1][]

        return trajectory, backward_probs_mat, forward_probs_mat

    def get_matrices_from_state(self, state, return_size=False):
        l = torch.tensor(state.size()[0] / 2)
        size = (int(torch.sqrt(l)), int(torch.sqrt(l)))
        l = int(l)
        current_adjacency, current_clustering = torch.reshape(state[:l], size), torch.reshape(state[l:], size)
        if return_size:
            return current_adjacency, current_clustering, size
        return current_adjacency, current_clustering

    def get_state_from_matrices(self, current_adjacency, current_clustering):
        return torch.concat((current_adjacency.flatten(), current_clustering.flatten()))

    def assign_clusters_torch(self, adjacency_matrix_full):
        cluster_assignments = torch.zeros(adjacency_matrix_full.shape[0])
        cluster_order = torch.randperm(adjacency_matrix_full.shape[0])
        adjacency_matrix_current = np.zeros(adjacency_matrix_full.shape)
        flow_total = 0
        l = self.n_clusters
        return None

    def assign_clusters(self, adjacency_matrix_full, alpha=1):
        # alpha is in [0,1] and weights the contribution of the NN vs random
        if self.using_cuda:
            return self.assign_clusters_torch(adjacency_matrix_full=adjacency_matrix_full)
        # 0 means no cluster, so clusters are 1-indexed
        cluster_assignments = np.zeros(adjacency_matrix_full.shape[0])
        cluster_order = np.random.choice(range(adjacency_matrix_full.shape[0]), adjacency_matrix_full.shape[0],
                                         replace=False)
        adjacency_matrix_current = np.zeros(adjacency_matrix_full.shape)
        flow_total = 0

        for node_index in cluster_order:
            # put the node into the current adjacency matrix, add masking for nodes in network
            # Should be unnecessary, as we only sum in the relevant part of the matrix
            adjacency_matrix_current[node_index] = adjacency_matrix_full[node_index]
            adjacency_matrix_current[:, node_index] = adjacency_matrix_full[:, node_index]

            cluster_feature_list = []
            for cluster_to_test in range(self.n_clusters):
                cluster_assignments_temp = cluster_assignments.copy()
                cluster_assignments_temp[node_index] = cluster_to_test
                cluster_feature_list.append(self.cluster_features(adjacency_matrix_current, cluster_assignments_temp))
            logits = self.model_forward.forward(torch.tensor(cluster_feature_list))
            logits_softmax = self.softmax(logits)
            # Numpy complains about it not summing to 1 if no outer softmax, probably a numerical instability issue
            logits_weighted = self.softmax(
                alpha * logits_softmax + (1 - alpha) * self.softmax(torch.tensor(np.random.random(self.n_clusters))))

            cluster_assigned = np.random.choice(range(self.n_clusters), p=logits_weighted.detach().numpy())
            # Negative values in flow
            cluster_assignments[node_index] = cluster_assigned + 1
        flow_total += logits_softmax[cluster_assigned]

        return cluster_assignments, flow_total

    def cluster_features(self, adjacency_matrix_current, cluster_assignments):
        # Optimize by only looking above diagonal
        cluster_features = torch.zeros((self.n_clusters, self.n_clusters))
        for cluster_number1 in range(self.n_clusters):
            for cluster_number2 in range(self.n_clusters):
                mask1 = cluster_assignments == cluster_number1 + 1
                mask2 = cluster_assignments == cluster_number2 + 1
                rows1 = adjacency_matrix_current[mask1]
                rows2 = np.logical_and(rows1, mask2)
                cluster_features[cluster_number1, cluster_number2] = np.sum(rows2)

        return torch.flatten(cluster_features)


class MLP(nn.Module):
    def __init__(self, n_hidden, n_nodes, n_clusters, output_size=1, n_layers=3):
        super().__init__()
        input_size = int(n_nodes ** 2)
        self.n_clusters = n_clusters
        # Forward and backward layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        for k in range(n_layers - 1):
            self.layers.append(nn.Linear(prev_size, n_hidden))
            # layers.append(nn.Linear(prev_size, self.n_hidden))
            self.layers.append(nn.ReLU())
            prev_size = n_hidden
        self.layers.append(nn.Linear(prev_size, output_size))
        self.layers.append(nn.Softplus())

    # Define the forward function of the neural network
    def forward(self, X):
        x = X
        for layer in self.layers:
            x = layer(x)
        return x


class SimpleBackwardModel:
    def __init__(self):
        pass

    def forward(self, current_clustering):
        # X is a vector of the two matrices, so check what nodes are in (k) and return a vector full
        # of 1/k. Seems to work
        # l = torch.tensor(X.size()[0] / 2)
        # size = (int(torch.sqrt(l)), int(torch.sqrt(l)))
        # l = int(l)
        # current_adjacency, current_clustering = torch.reshape(X[:l], size), torch.reshape(X[l:], size)
        current_nodes = torch.sum(current_clustering, dim=1) > 0
        return current_nodes / torch.sum(current_nodes)


if __name__ == '__main__':
    import torch

    using_cuda = False
    if torch.cuda.is_available() and using_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = GraphNet(using_cuda=using_cuda)
    a = torch.ones((5, 5))
    a[0, 1], a[1, 0] = 0, 0
    a[0, 3], a[3, 0] = 0, 0
    a[0, 4], a[4, 0] = 0, 0

    # print(net.assign_clusters(a))
    b = torch.concat((torch.tensor(a).flatten(), torch.tensor(a).flatten()))
    bas = SimpleBackwardModel()
    print(net.sample_trajectory_backwards(b))
