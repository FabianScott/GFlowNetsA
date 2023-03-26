import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import itertools


class GraphNet:
    def __init__(self,
                 n_nodes,
                 termination_chance=None,
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
        self.n_nodes = n_nodes
        self.size = (n_nodes, n_nodes)
        self.model_forward = MLP(output_size=1,
                                 n_nodes=n_nodes,
                                 n_hidden=n_hidden,
                                 n_clusters=n_clusters,
                                 n_layers=n_layers)

        self.model_backward = MLP(output_size=int(n_nodes),
                                  n_nodes=n_nodes,
                                  n_hidden=n_hidden,
                                  n_clusters=n_clusters,
                                  n_layers=n_layers) if using_backward_model \
            else SimpleBackwardModel()
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=0)
        self.z0 = nn.Parameter(torch.tensor([.0]))
        self.optimizer = torch.optim.Adam(itertools.chain(self.model_forward.parameters(), (self.z0,)), lr=self.lr)
        self.using_cuda = using_cuda
        self.using_backward_model = using_backward_model
        self.state_length = 2 * self.n_nodes ** 2 + self.n_nodes
        self.termination_chance = 1/n_nodes if termination_chance is None else termination_chance

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
        for epoch in tqdm(range(epochs)):
            for i in range(0, X.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x = X[indices]

                outputs = []
                targets = []
                for x in batch_x:
                    _, forward, backward = self.calculate_flows_from_terminal_state(x)
                    outputs.append(forward)
                    targets.append(backward)

                loss = self.mse_loss(outputs, targets)

                loss.backward()
                self.optimizer.step()

    def trajectory_balance_loss(self, final_states):
        # final_states is a list of vectors representing states with at minimum one node missing
        losses = []

        for state in final_states:
            # The probabilities are computed while creating the trajectories
            trajectory, probs_log_sum_backward, probs_log_sum_forward = self.sample_trajectory_backwards(state)
            loss = self.mse_loss(probs_log_sum_forward, probs_log_sum_backward)
            losses.append(loss)

        return losses

    def calculate_flows_from_terminal_state(self, terminal_state):
        # The terminal_state does encode information about what node was most recently clustered
        current_adjacency, current_clustering, last_node_placed = self.get_matrices_from_state(terminal_state) # ignores the one-hot part
        current_clustering_list, number_of_clusters = self.get_clustering_list(current_clustering)
        prob_log_sum_forward = torch.zeros(1)
        # Need IRM to add here, leave out for now
        prob_log_sum_backward = torch.zeros(1)
        # Will track the order in which nodes are removed from the graph
        trajectory = torch.zeros(self.n_nodes)
        node_no = 1

        while torch.sum(current_clustering_list).item():
            probs_backward = self.model_backward.forward(
                current_clustering)  # should take entire state when real model is used
            index_chosen = torch.multinomial(probs_backward, 1)
            prob_log_sum_backward += torch.log(probs_backward[index_chosen])
            # Remove the node from the clustering
            current_clustering[index_chosen] = 0
            current_clustering[:, index_chosen] = 0
            # Get the forward flow
            last_node_placed = torch.zeros(self.n_nodes)
            last_node_placed[index_chosen] = 1
            current_state = torch.concat((current_adjacency.flatten(), current_clustering.flatten(), last_node_placed))
            forward_probs = self.forward_flow(current_state)
            # Cluster labels are 1-indexed
            cluster_origin_index = current_clustering_list[index_chosen] - 1
            # Now calculate the forward flow into the state we passed to the backward model:
            current_clustering_list, _ = self.get_clustering_list(current_clustering)
            node_index_forward = torch.sum(current_clustering_list[:index_chosen] == 0)
            prob_log_sum_forward += torch.log(forward_probs[(int(node_index_forward), int(cluster_origin_index[0]))])
            trajectory[index_chosen] = node_no
            node_no += 1

        return trajectory, prob_log_sum_forward, prob_log_sum_backward + self.z0

    def sample_trajectory_backwards(self, state):
        # state is the (k ** 2) * 2 long vector of both matrices
        # reshape stuff to make it easier to work with

        current_adjacency, current_clustering, _ = self.get_matrices_from_state(state)
        # zeros = torch.zeros(self.size[0])
        # zeros_ = torch.zeros((self.size[0], 1))

        trajectory, backward_probs_mat, forward_probs_mat = torch.zeros(self.size), torch.zeros(self.size), torch.zeros(
            (self.size[0], self.size[0] ** 2))
        prob_sum_forward, prob_sum_backward = torch.zeros(1), torch.zeros(1)
        probs_backward = self.model_backward.forward(current_clustering)

        for i in range(self.size[0]):
            # put the backward probabilities in at the start of the loop to avoid the last one
            backward_probs_mat[i] = probs_backward

            index_chosen = torch.multinomial(probs_backward, 1)
            # remove the chosen node from the clustering
            current_clustering[index_chosen] = 0
            current_clustering[:, index_chosen] = 0

            prob_sum_backward += torch.log(torch.tensor(probs_backward[index_chosen]))
            probs_backward = self.model_backward.forward(current_clustering)

            # put the forward probabilities in at the end to skip the first iteration but include the last
            current_state = self.get_state_from_matrices(current_adjacency, current_clustering)
            probs_forward = self.model_forward.forward(current_state).flatten()

            # forward_probs_mat[i] = probs_forward
            prob_sum_forward += torch.log(torch.tensor(probs_forward[index_chosen]))
            trajectory[i][index_chosen] = 1
        # trajectory[i + 1][]

        return trajectory, prob_sum_backward, prob_sum_forward

    def forward_flow(self, state):
        # return the forward flow for all possible actions (choosing node, assigning cluster)
        # using the idea that the NN rates each possible future state. Variable output size!!
        current_adjacency, current_clustering, last_node_placed = self.get_matrices_from_state(state)
        # number of clusters starts at 1
        clustering_list, number_of_clusters = self.get_clustering_list(current_clustering)
        nodes_to_place = torch.argwhere(torch.sum(current_clustering, dim=0) == 0)  # it's symmetrical

        output = torch.zeros((nodes_to_place.size()[0], number_of_clusters))
        possible_node = 0
        for node_index in nodes_to_place:
            # node index is a tensor
            for possible_cluster in range(1, number_of_clusters + 1):
                temp_clustering_list = torch.clone(clustering_list)
                temp_clustering_list[node_index] = possible_cluster
                temp_clustering = self.place_clustering_from_list(temp_clustering_list, number_of_clusters)
                temp_state = torch.concat(
                    (current_adjacency.flatten(), temp_clustering.flatten(), last_node_placed))

                output[possible_node, possible_cluster - 1] = self.model_forward.forward(temp_state)
            possible_node += 1
        return output

    def sample_forward(self, current_adjacency, epochs=None):
        # Not Working Yet!!
        if epochs is None:
            epochs = self.epochs

        final_states = torch.zeros((epochs, self.state_length))

        for epoch in tqdm(range(epochs)):
            # Initialize the empty clustering and one-hot vector
            current_clustering = torch.zeros(self.size)
            last_node_placed = torch.zeros(self.n_nodes)
            # Here to ensure an empty state if
            # Each iteration has self.termination_chance of being the last, and we ensure no empty states
            while torch.rand(1) > self.termination_chance or not last_node_placed.sum():
                # Create the state vector and pass it to the NN
                current_state = torch.concat(
                    (current_adjacency.flatten(), current_clustering.flatten(), last_node_placed))
                current_clustering_list, num_clusters = self.get_clustering_list(current_clustering)
                # Reset last node placed after use
                last_node_placed = torch.zeros(self.n_nodes)
                forward_flows = self.forward_flow(current_state)
                forward_probs = self.softmax_matrix(forward_flows).flatten()
                # Sample from the output and retrieve the indices of the chosen node and clustering
                index_chosen = torch.multinomial(forward_probs, 1)
                # First find the row and column we have chosen from the probs
                node_chosen = index_chosen // num_clusters
                cluster_index_chosen = index_chosen - node_chosen
                # Next, increment the node chosen by the number of nodes ahead of it that were already in the graph
                node_chosen += torch.sum(current_clustering_list[:node_chosen] > 0)
                # Update the cluster
                # Labels are 1-indexed, indices are 0 indexed
                current_clustering_list[node_chosen] = torch.tensor(cluster_index_chosen + 1, dtype=torch.float32)
                current_clustering = self.place_clustering_from_list(current_clustering_list, num_clusters)
                last_node_placed[node_chosen] = 1

            final_states[epoch] = current_state

        return final_states

#%% Helpers:
    def place_clustering_from_list(self, clustering_list, number_of_clusters):
        # Returns the clustering matrix associated with the clustering list, takes the number of
        # cluster as an argument because it has been calculated prior to the use of this function
        current_clustering = torch.zeros(self.size)
        for cluster_no in range(1, number_of_clusters + 1):
            # Find all indices belonging to the current cluster we're looking at
            cluster_positions = torch.argwhere(clustering_list == cluster_no).flatten()
            # Find the unique combination of these indices, including the diagonal elements (probably not the most efficient)
            indices = torch.unique(torch.combinations(torch.concat((cluster_positions, cluster_positions)), r=2), dim=0)
            for ind in indices:
                current_clustering[(ind[0], ind[1])] = 1
                # current_clustering[(ind[1], ind[0])] = 1 # Left in case needed when changing for efficiency
        return current_clustering

    def get_clustering_list(self, current_clustering):
        # return a 1-dimensional tensor of what cluster each node is in

        # zeros = torch.zeros(self.size[0])
        # zeros_ = torch.zeros((self.size[0], 1))

        current_clustering_copy = torch.clone(current_clustering)
        output = torch.zeros(current_clustering_copy.size()[0])
        cluster_no = 1
        node_no = 0
        while torch.sum(current_clustering_copy):
            row = current_clustering_copy[node_no]
            if torch.sum(row):
                indices = torch.argwhere(row)
                output[indices] = cluster_no
                current_clustering_copy[indices] = 0
                current_clustering_copy[:, indices] = 0
                cluster_no += 1
            node_no += 1

        return output, cluster_no

    def get_matrices_from_state(self, state):
        # returns the adjacency matrix, clustering matrix and one-hot vector of the last node placed into the graph
        l = self.n_nodes ** 2
        size = self.size
        current_adjacency, current_clustering = torch.reshape(state[:l], size), torch.reshape(state[l:2 * l], size)
        return current_adjacency, current_clustering, state[2*l:]

    def get_state_from_matrices(self, current_adjacency, current_clustering):
        return torch.concat((current_adjacency.flatten(), current_clustering.flatten()))

    def softmax_matrix(self, inp):
        return torch.exp(inp)/torch.sum(torch.exp(inp))

class MLP(nn.Module):
    def __init__(self, n_hidden, n_nodes, n_clusters, output_size=1, n_layers=3):
        super().__init__()
        # takes adj_mat, clustering_mat, one-hot node_placed
        input_size = int(2 * n_nodes ** 2 + n_nodes)
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
        # To ensure positive output
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
        # state is adj_mat and clustering_mat
        current_nodes = torch.sum(current_clustering, dim=1) > 0
        return current_nodes / torch.sum(current_nodes)


if __name__ == '__main__':
    import torch

    using_cuda = False
    if torch.cuda.is_available() and using_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    a = torch.ones((5, 5))
    a[0, 1], a[1, 0] = 0, 0
    a[0, 3], a[3, 0] = 0, 0
    a[0, 4], a[4, 0] = 0, 0

    net = GraphNet(n_nodes=a.size()[0], using_cuda=using_cuda)

    # print(net.assign_clusters(a))
    bas = SimpleBackwardModel()
    #
    cluster_list = torch.tensor([1, 1, 2, 3, 0])
    clustering_mat = net.place_clustering_from_list(cluster_list, 4)
    one_hot_node = torch.zeros(5)
    one_hot_node[3] = 1
    b = torch.concat((torch.tensor(a).flatten(), torch.tensor(clustering_mat).flatten(), one_hot_node))
    print(net.forward_flow(b))
    print(net.calculate_flows_from_terminal_state(b))
    final_states = net.sample_forward(a, epochs=10)
    for state in final_states:
        print(net.get_matrices_from_state(state)[1 ])
