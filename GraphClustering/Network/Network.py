import numpy as np
import torch
import torch.nn as nn


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
                 using_cuda=False
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
        self.model_forward = MLP(n_nodes=n_nodes,
                                 n_hidden=n_hidden,
                                 n_clusters=n_clusters,
                                 n_layers=n_layers)
        self.model_backward = MLP(n_nodes=n_nodes,
                                  n_hidden=n_hidden,
                                  n_clusters=n_clusters,
                                  n_layers=n_layers)
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=0)
        self.optimizer = torch.optim.Adam(self.model_forward.parameters(), lr=self.lr)
        self.using_cuda = using_cuda

    def create_model(self):
        # Define the layers of the neural network
        # layers = []
        # Assuming the features extracted for each cluster has size 1

        # Create an instance of the neural network and return it
        # net = nn.Module()
        return MLP(n_hidden=self.n_hidden, n_clusters=self.n_clusters, n_layers=self.n_layers, output_size=1)

    def train(self, X, Y, epochs=100, batch_size=None):
        # X: an iterable/index-able of final cluster assignments
        # Y: an iterable/index-able of IRM values for each X
        if batch_size is None:
            batch_size = self.batch_size

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

    def assign_clusters_torch(self, adjacency_matrix_full):
        cluster_assignments = torch.zeros(adjacency_matrix_full.shape[0])
        cluster_order = np.random.choice(range(adjacency_matrix_full.shape[0]), adjacency_matrix_full.shape[0], replace=False)
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
        cluster_order = np.random.choice(range(adjacency_matrix_full.shape[0]), adjacency_matrix_full.shape[0], replace=False)
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
            logits = self.model_forward.forward(cluster_feature_list)
            logits_softmax = self.softmax(logits)
            # Numpy complains about it not summing to 1 if no outer softmax, probably a numerical instability issue
            logits_weighted = self.softmax(alpha * logits_softmax + (1 - alpha) * self.softmax(torch.tensor(np.random.random(self.n_clusters))))

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
        input_size = int(n_nodes**2)
        self.n_clusters = n_clusters
        # Forward and backward layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        for k in range(n_layers-1):
            self.layers.append(nn.Linear(prev_size, n_hidden))
            # layers.append(nn.Linear(prev_size, self.n_hidden))
            self.layers.append(nn.ReLU())
            prev_size = n_hidden
        self.layers.append(nn.Linear(prev_size, output_size))
        self.layers.append(nn.Softplus())

    # Define the forward function of the neural network
    def forward_(self, X):
        # Given an iterable of adjacency matrices as tensors for each possible action to take
        flows = torch.Tensor(np.zeros(self.n_clusters))
        for i, x in enumerate(X):
            for layer in self.layers:
                x = layer(x)
            flows[i] = x[0]
        return flows

    def forward(self, X):
        x = X
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    import torch

    using_cuda = False
    if torch.cuda.is_available() and using_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    net = GraphNet(using_cuda=using_cuda)
    a = np.ones((5,5))
    a[0,1], a[1,0] = 0, 0
    a[0,3], a[3,0] = 0, 0
    a[0,4], a[4,0] = 0, 0

    print(net.assign_clusters(a))
