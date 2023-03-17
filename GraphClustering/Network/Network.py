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
                 n_clusters=4
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
        self.model = self.create_model()
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def create_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Define the layers of the neural network
        # layers = []
        # Assuming the features extracted for each cluster has size 1

        # Create an instance of the neural network and return it
        # net = nn.Module()
        return MLP(n_hidden=self.n_hidden, n_clusters=self.n_clusters, n_layers=self.n_layers, output_size=1)

    def train(self, X, Y, epochs=100, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        permutation = torch.randperm(X.size()[0])
        for epoch in range(epochs):
            for i in range(0, X.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X[indices], Y[indices]

                outputs = self.model.forward(batch_x)
                loss = self.mse_loss(outputs, batch_y)

                loss.backward()
                self.optimizer.step()

    def assign_clusters(self, adjacency_matrix_full):
        cluster_assignments = np.zeros(adjacency_matrix_full.shape[0])
        cluster_order = np.random.choice(range(adjacency_matrix_full.shape[0]), adjacency_matrix_full.shape[0], replace=False)
        adjacency_matrix_current = np.zeros(adjacency_matrix_full.shape)
        flow_total = 0

        for node_index in cluster_order:
            # put the node into the current adjacency matrix, add masking for nodes in network
            adjacency_matrix_current[node_index] = adjacency_matrix_full[node_index]
            adjacency_matrix_current[:, node_index] = adjacency_matrix_full[:, node_index]

            cluster_feature_list = []
            for cluster_to_test in range(self.n_clusters):
                cluster_assignments_temp = cluster_assignments.copy()
                cluster_assignments_temp[node_index] = cluster_to_test
                cluster_feature_list.append(self.cluster_features(adjacency_matrix_current, cluster_assignments_temp))
            logits = self.model.forward(cluster_feature_list)
            logits_softmax = self.softmax(logits)
            cluster_assigned = np.random.choice(range(self.n_clusters), p=logits_softmax.detach().numpy())
            # Negative values in flow
            flow_total += logits_softmax[cluster_assigned]
            cluster_assignments[node_index] = cluster_assigned

        return cluster_assignments, flow_total

    def cluster_features(self, adjacency_matrix_current, cluster_assignments):
        # Optimize by only looking above diagonal
        cluster_features = torch.zeros((self.n_clusters, self.n_clusters))
        for cluster_number1 in range(self.n_clusters):
            for cluster_number2 in range(self.n_clusters):
                mask1 = cluster_assignments == cluster_number1 + 1
                mask2 = cluster_assignments == cluster_number2 + 1
                rows1 = adjacency_matrix_current[mask1]
                rows2 = rows1 @ mask2
                cluster_features[cluster_number1, cluster_number2] = np.sum(rows2)

        return torch.flatten(cluster_features)


class MLP(nn.Module):
    def __init__(self, n_hidden, n_clusters, output_size=1, n_layers=3):
        super().__init__()
        input_size = int(n_clusters**2)
        self.n_clusters = n_clusters
        # Forward and backward layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        for k in range(n_layers):
            self.layers.append(nn.Linear(prev_size, n_hidden))
            # layers.append(nn.Linear(prev_size, self.n_hidden))
            self.layers.append(nn.ReLU())
            prev_size = n_hidden
        self.layers.append(nn.Linear(prev_size, 1))

    # Define the forward function of the neural network
    def forward(self, X):
        # Given an iterable of adjacency matrices as tensors for each possible action to take
        flows = torch.Tensor(np.zeros(self.n_clusters))
        for i, x in enumerate(X):
            for layer in self.layers:
                x = layer(x)
            flows[i] = x[0]
        return flows


if __name__ == '__main__':
    import torch

    net = GraphNet()
    a = np.ones((5,5))
    a[0,1], a[1,0] = 0, 0
    a[0,3], a[3,0] = 0, 0
    a[0,4], a[4,0] = 0, 0

    net.assign_clusters(a)
