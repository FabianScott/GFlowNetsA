import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import itertools
from GraphClustering.IRM_post import torch_posterior, p_x_giv_z


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

    def train(self, X, Y=None, epochs=100, batch_size=None):
        # X: an iterable/index-able of final cluster assignments
        # Y: an iterable/index-able of IRM values for each X
        if batch_size is None:
            batch_size = self.batch_size
        # if complete_graphs is None:
        #     print(f'Missing iterable indicating which graphs can be evaluated using IRM!')
        #     raise NotImplementedError

        permutation = torch.randperm(X.size()[0])
        for epoch in tqdm(range(epochs)):
            for i in range(0, X.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x = X[indices]

                outputs = torch.zeros((batch_size, 1))
                targets = torch.zeros((batch_size, 1))
                for i, x in enumerate(batch_x):

                    _, forward, backward = self.log_sum_flows(x)
                    outputs[i] = forward
                    if self.is_terminal(x):
                        # Should calculate IRM value of the state:
                        adjacency_matrix, clustering_matrix, _ = self.get_matrices_from_state(x)
                        backward = torch_posterior(adjacency_matrix, clustering_matrix)
                    targets[i] = backward

                loss = self.mse_loss(outputs, targets)

                loss.backward()
                self.optimizer.step()

    # def trajectory_balance_loss(self, final_states):
    #     """
    #     Using the log_sum_flows function, create a list of
    #     losses to pass to the optimizer.
    #     OBSOLETE FUNCTION!!
    #     :param final_states: iterable of state vectors
    #     :return: losses: list of losses
    #     """
    #     # final_states is a list of vectors representing states with at minimum one node missing
    #     losses = []
    #
    #     for state in final_states:
    #         # The probabilities are computed while creating the trajectories
    #         trajectory, probs_log_sum_backward, probs_log_sum_forward = self.log_sum_flows(state)
    #         loss = self.mse_loss(probs_log_sum_forward, probs_log_sum_backward)
    #         losses.append(loss)
    #
    #     return losses

    def log_sum_flows(self, terminal_state):
        """
        Given a terminal state, calculate the log sum of the flow
        probabilities backwards and forward along with the trajectory
        indicating when what node was removed by sampling from the
        backward model.
        :param terminal_state:
        :return: trajectory (n_nodes, n_nodes),
                forward_log_sum_probs (int), backward_log_sum_probs (int)
        """
        # The terminal_state does encode information about what node was most recently clustered
        adjacency_matrix, clustering_matrix, last_node_placed = self.get_matrices_from_state(terminal_state) # ignores the one-hot part
        current_clustering_list, number_of_clusters = self.get_clustering_list(clustering_matrix)
        prob_log_sum_forward = torch.zeros(1)
        # Need IRM to add here, leave out for now
        prob_log_sum_backward = torch.zeros(1)
        # Will track the order in which nodes are removed from the graph
        trajectory = torch.zeros(self.n_nodes)
        node_no = 1

        while torch.sum(current_clustering_list).item():
            probs_backward = self.model_backward.forward(
                clustering_matrix)  # should take entire state when real model is used
            index_chosen = torch.multinomial(probs_backward, 1)
            prob_log_sum_backward += torch.log(probs_backward[index_chosen])
            # Remove the node from the clustering
            clustering_matrix[index_chosen] = 0
            clustering_matrix[:, index_chosen] = 0
            # Get the forward flow
            last_node_placed = torch.zeros(self.n_nodes)
            last_node_placed[index_chosen] = 1
            current_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten(), last_node_placed))
            forward_probs = self.forward_flow(current_state)
            # Cluster labels are 1-indexed
            cluster_origin_index = current_clustering_list[index_chosen] - 1
            # Now calculate the forward flow into the state we passed to the backward model:
            current_clustering_list, _ = self.get_clustering_list(clustering_matrix)
            node_index_forward = torch.sum(current_clustering_list[:index_chosen] == 0)
            prob_log_sum_forward += torch.log(forward_probs[(int(node_index_forward), int(cluster_origin_index[0]))])
            trajectory[index_chosen] = node_no
            node_no += 1

        return trajectory, prob_log_sum_forward, prob_log_sum_backward + self.z0

    def forward_flow(self, state):
        """
        Given a state, return the forward flow from the current
        forward model.
        :param state: vector (n_nodes,)
        :return:
        """
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
                temp_clustering = self.get_clustering_matrix(temp_clustering_list, number_of_clusters)
                temp_state = torch.concat(
                    (current_adjacency.flatten(), temp_clustering.flatten(), last_node_placed))

                output[possible_node, possible_cluster - 1] = self.model_forward.forward(temp_state)
            possible_node += 1
        return output

    def sample_forward(self, adjacency_matrix, epochs=None):
        """
        Given an adjacency matrix, cluster some graphs and return
        epochs number of final states reached using the current
        forward model. If epochs is left as None use self.epochs.
        :param adjacency_matrix: matrix (n_nodes, n_nodes)
        :param epochs: (None or int)
        :return:
        """

        if epochs is None:
            epochs = self.epochs

        final_states = torch.zeros((epochs, self.state_length))

        for epoch in tqdm(range(epochs)):
            # Initialize the empty clustering and one-hot vector
            clustering_matrix = torch.zeros(self.size)
            last_node_placed = torch.zeros(self.n_nodes)
            clustering_list = torch.zeros(self.n_nodes)
            # Here to ensure an empty state if
            # Each iteration has self.termination_chance of being the last, and we ensure no empty states
            while not last_node_placed.sum() or (torch.rand(1) >= self.termination_chance and torch.sum(clustering_list > 0) != self.n_nodes):
                # Create the state vector and pass it to the NN
                current_state = torch.concat(
                    (adjacency_matrix.flatten(), clustering_matrix.flatten(), last_node_placed))
                clustering_list, num_clusters = self.get_clustering_list(clustering_matrix)
                # Reset last node placed after use
                last_node_placed = torch.zeros(self.n_nodes)
                forward_flows = self.forward_flow(current_state)
                forward_probs = self.softmax_matrix(forward_flows).flatten()
                # Sample from the output and retrieve the indices of the chosen node and clustering
                index_chosen = torch.multinomial(forward_probs, 1)
                # First find the row and column we have chosen from the probs
                node_chosen = index_chosen // num_clusters
                cluster_index_chosen = int(node_chosen * num_clusters - index_chosen)
                # Next, increment the node chosen by the number of nodes ahead of it that were already in the graph
                node_chosen += torch.sum(clustering_list[:node_chosen] > 0)
                # Update the cluster
                # Labels are 1-indexed, indices are 0 indexed
                clustering_list[node_chosen] = torch.tensor(cluster_index_chosen + 1, dtype=torch.float32)
                clustering_matrix = self.get_clustering_matrix(clustering_list, num_clusters)
                last_node_placed[node_chosen] = 1

            final_states[epoch] = current_state

        return final_states

#%% Helpers:
    def get_clustering_matrix(self, clustering_list, number_of_clusters):
        """
        Reverse of get clustering list, given the clustering list
        returns the clustering matrix
        :param clustering_list: vector (n_nodes,)
        :param number_of_clusters: int
        :return: clustering_matrix (n_nodes, n_nodes)
        """
        # Returns the clustering matrix associated with the clustering list, takes the number of
        # cluster as an argument because it has been calculated prior to the use of this function
        clustering_matrix = torch.zeros(self.size)
        for cluster_no in range(1, number_of_clusters + 1):
            # Find all indices belonging to the current cluster we're looking at
            cluster_positions = torch.argwhere(clustering_list == cluster_no).flatten()
            # Find the unique combination of these indices, including the diagonal elements (probably not the most efficient)
            indices = torch.unique(torch.combinations(torch.concat((cluster_positions, cluster_positions)), r=2), dim=0)
            for ind in indices:
                clustering_matrix[(ind[0], ind[1])] = 1
                # clustering_matrix[(ind[1], ind[0])] = 1 # Left in case needed when changing for efficiency
        return clustering_matrix

    def get_clustering_list(self, clustering_matrix):
        """
        Given a clustering matrix, returns a clustering list and
        the number of clusters
        :param clustering_matrix: matrix (n_nodes, n_nodes)
        :return: clustering_list (n_nodes,), number_of_clusters (int)
        """
        # return a 1-dimensional tensor of what cluster each node is in

        # zeros = torch.zeros(self.size[0])
        # zeros_ = torch.zeros((self.size[0], 1))

        current_clustering_copy = torch.clone(clustering_matrix)
        clustering_list = torch.zeros(current_clustering_copy.size()[0])
        number_of_clusters = 1
        node_no = 0
        while torch.sum(current_clustering_copy):
            row = current_clustering_copy[node_no]
            if torch.sum(row):
                indices = torch.argwhere(row)
                clustering_list[indices] = number_of_clusters
                current_clustering_copy[indices] = 0
                current_clustering_copy[:, indices] = 0
                number_of_clusters += 1
            node_no += 1

        return clustering_list, number_of_clusters

    def get_matrices_from_state(self, state):
        """
        Given a state vector of the size the class is initialized with, return the
        different parts as matrices and vectors
        :param state: vector (2 * n_nodes ^ 2 + n_nodes)
        :return: adjacency matrix and clustering matrix (n_nodes, n_nodes), last node placed (n_nodes,)
        """
        # returns the adjacency matrix, clustering matrix and one-hot vector of the last node placed into the graph
        l = self.n_nodes ** 2
        size = self.size
        current_adjacency, current_clustering = torch.reshape(state[:l], size), torch.reshape(state[l:2 * l], size)
        return current_adjacency, current_clustering, state[2*l:]

    def softmax_matrix(self, inp):
        """
        Takes the softmax across an entire matrix/vector of *any* size
        :param inp:
        :return:
        """
        return torch.exp(inp)/torch.sum(torch.exp(inp))

    def is_terminal(self, state):
        """
        Given a state vector, check if it is in a terminal state
        by summing the diagonal of the clustering matrix
        :param state:
        :return:
        """
        # Check the diagonal of the clustering matrix using indexing and if the sum == n_nodes it is terminal

        return state[::self.n_nodes][self.n_nodes ** 2:2 * (self.n_nodes ** 2)].sum() == self.n_nodes


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
    adjacency_matrix = torch.ones((5, 5))
    adjacency_matrix[0, 1], adjacency_matrix[1, 0] = 0, 0
    adjacency_matrix[0, 3], adjacency_matrix[3, 0] = 0, 0
    adjacency_matrix[0, 4], adjacency_matrix[4, 0] = 0, 0

    net = GraphNet(n_nodes=adjacency_matrix.size()[0], using_cuda=using_cuda)

    # print(net.assign_clusters(a))
    bas = SimpleBackwardModel()
    #
    clustering_list = torch.tensor([1, 1, 2, 3, 0])
    clustering_mat = net.get_clustering_matrix(clustering_list, 4)
    last_node_placed = torch.zeros(5)
    last_node_placed[3] = 1
    b = torch.concat((torch.tensor(adjacency_matrix).flatten(), torch.tensor(clustering_mat).flatten(), last_node_placed))
    # print(net.forward_flow(b))
    # print(net.log_sum_flows(b))
    # final_states = net.sample_forward(a, epochs=10)
    # for state in final_states:
    #     print(net.get_matrices_from_state(state)[1])
    # net.train(final_states)
    print(p_x_giv_z(adjacency_matrix.detach().numpy(), clustering_list.detach().numpy()))
    print(torch_posterior(adjacency_matrix, clustering_list))
    print(net.forward_flow(b))
    net.termination_chance = 0
    X = net.sample_forward(adjacency_matrix=adjacency_matrix)
    net.train(X)
    X1 = net.sample_forward(adjacency_matrix=adjacency_matrix)

    print(net.forward_flow(b))
