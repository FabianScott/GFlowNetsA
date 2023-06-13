import pandas as pd
import torch
import itertools
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from scipy import stats
from scipy.special import betaln, gammaln, logsumexp
from torch.special import gammaln as torch_gammaln
from torch.distributions import Beta
import matplotlib.pyplot as plt
from collections import defaultdict


class GraphNet:
    def __init__(self,
                 n_nodes,
                 a=1.,
                 b=1.,
                 A_alpha=1.,
                 termination_chance=None,
                 env=None,
                 n_layers=5,
                 n_hidden=64,
                 gamma=0.5,
                 epochs=100,
                 lr=0.005,
                 # decay_steps=10000,
                 # decay_rate=0.8,
                 n_samples=1000,
                 batch_size=10,
                 n_clusters=4,
                 using_cuda=False,
                 using_backward_model=False,
                 using_direct_loss=True
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
        self.a, self.b, self.A_alpha = torch.tensor([a]), torch.tensor([b]), torch.tensor([A_alpha])
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
                                  n_layers=n_layers,
                                  softmax=True) if using_backward_model \
            else SimpleBackwardModel()
        self.mse_loss = nn.MSELoss()
        self.softmax = torch.nn.Softmax(dim=0)
        self.z0 = nn.Parameter(torch.tensor([.0]))
        chain = itertools.chain(self.model_forward.parameters(), self.model_backward.parameters(), (self.z0,)) \
            if using_backward_model else itertools.chain(self.model_forward.parameters(), (self.z0,))
        self.optimizer = torch.optim.Adam(chain, lr=self.lr)
        self.using_cuda = using_cuda
        self.using_backward_model = using_backward_model
        self.using_direct_loss = using_direct_loss
        self.state_length = 2 * self.n_nodes ** 2
        self.termination_chance = 0 if termination_chance is None else termination_chance

    def create_model(self):
        """
        Create an instance of the neural network and return it
        :return:
        """
        return MLP(n_hidden=self.n_hidden, n_clusters=self.n_clusters, n_layers=self.n_layers, output_size=1)

    def train(self, X, Y=None, epochs=100, batch_size=None, verbose=False):
        """
        Given an iterable of final states and a number of epochs, train the
        network.
        :param X:
        :param Y:
        :param epochs:
        :param batch_size:
        :return:
        """
        # X: an iterable/index-able of final cluster assignments
        # Y: an iterable/index-able of IRM values for each X
        if batch_size is None:
            batch_size = self.batch_size
        # if complete_graphs is None:
        #     print(f'Missing iterable indicating which graphs can be evaluated using IRM!')
        #     raise NotImplementedError
        losses = torch.zeros(epochs)
        for epoch in tqdm(range(epochs), desc='Training'):
            # Permute every epoch
            permutation = torch.randperm(X.size()[0])
            loss_this_epoch = 0
            # Train on every batch of the training data
            for i in range(0, X.size()[0], batch_size):
                # Extract the batch
                indices = permutation[i:i + batch_size]
                batch_x = X[indices]
                # Initialize tensors for the gradient step
                outputs = torch.zeros((batch_size, 1))
                targets = torch.zeros((batch_size, 1))

                for j, x in enumerate(batch_x):
                    # Get the forward and backward flows from the state
                    # Reward = 0
                    if self.is_terminal(x):
                        adjacency_matrix, clustering_matrix = self.get_matrices_from_state(x)
                        # Subtract 1 from the clustering_list to make it 0-indexed for the posterior function
                        clustering_list = self.get_clustering_list(clustering_matrix)[0] - 1
                        # Calculates IRM value of the state:
                        Reward = torch_posterior(adjacency_matrix, clustering_list, a=self.a, b=self.b,
                                                 A_alpha=self.A_alpha)

                    # Save time!!!
                    if self.using_direct_loss:
                        outflow_network = self.model_forward.forward(x)
                        outputs[j] = outflow_network
                        targets[j] = Reward
                    else:
                        _, forward, backward = self.log_sum_flows(x)
                        outputs[j] = forward
                        targets[j] = Reward + backward

                loss = self.mse_loss(outputs, targets)
                loss_this_epoch += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            losses[epoch] = loss_this_epoch
            if verbose:
                print(f'Loss at iteration {epoch + 1}:\t{loss_this_epoch}')
        return losses

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
        adjacency_matrix, clustering_matrix = self.get_matrices_from_state(terminal_state)
        # To prevent changes to the original values??
        adjacency_matrix, clustering_matrix = adjacency_matrix.clone(), clustering_matrix.clone()
        current_clustering_list, number_of_clusters = self.get_clustering_list(clustering_matrix)
        prob_log_sum_forward = torch.zeros(1)
        # Need IRM to add here, leave out for now
        prob_log_sum_backward = torch.zeros(1)
        # Will track the order in which nodes are removed from the graph
        trajectory = torch.zeros(self.n_nodes)
        node_no = 1
        current_state = terminal_state

        while torch.sum(current_clustering_list).item():
            probs_backward = self.model_backward.forward(current_state) if self.using_backward_model \
                else self.model_backward.forward(clustering_matrix)
            index_chosen = torch.multinomial(probs_backward, 1)
            prob_log_sum_backward += torch.log(probs_backward[index_chosen])
            # Remove the node from the clustering
            clustering_matrix[index_chosen] = 0
            clustering_matrix[:, index_chosen] = 0
            # Get the forward flow
            current_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))
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
        :param state: vector ((n_nodes^2) * 2,)
        :return:
        """
        # return the forward flow for all possible actions (choosing node, assigning cluster)
        # using the idea that the NN rates each possible future state. Variable output size!!
        current_adjacency, current_clustering = self.get_matrices_from_state(state)
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
                    (current_adjacency.flatten(), temp_clustering.flatten()))

                output[possible_node, possible_cluster - 1] = self.model_forward.forward(temp_state)
            possible_node += 1
        # assert not any((torch.isinf(output).flatten() + torch.isnan(output).flatten()))
        return output

    # def get_all_probs(self, adjacency_matrix):
    #     """
    #     OBSOLETE!!
    #     :param adjacency_matrix:
    #     :return: list of dictionaries
    #     """
    #     from copy import deepcopy
    #     from collections import defaultdict
    #
    #     def return_0():
    #         return torch.zeros(1)
    #
    #     clustering_matrix = torch.zeros(adjacency_matrix.size())
    #     previous_states = [torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))]
    #     # The probability of starting in the staring state is 1
    #     previous_layer_dict = {previous_states[0]: torch.log(torch.ones(1))}
    #     out = []
    #     # Search through the entire space
    #     for _ in range(self.n_nodes):
    #         current_layer_dict = defaultdict(return_0)
    #         current_states_temp = []
    #         # Loop through all states from the previous layer
    #         for previous_state in tqdm(previous_states):
    #             # Get the forward flows and convert into probabilities
    #             forward_logits = self.forward_flow(previous_state)
    #             forward_probs = self.softmax_matrix(forward_logits).flatten()
    #             # Loop through all potential actions and store each probability
    #             for index_chosen, prob in enumerate(forward_probs):
    #                 # Calculate the log probability
    #                 log_prob = torch.log(torch.tensor(prob))
    #                 new_state, clustering_list = self.place_node(previous_state, index_chosen,
    #                                                              return_clustering_list=True)
    #                 # Use the clustering list as the keys in the dictionary to save space
    #                 current_states_temp.append(new_state)
    #                 current_layer_dict[clustering_list] = torch.logaddexp(current_layer_dict[clustering_list],
    #                                                                       log_prob + previous_layer_dict[
    #                                                                           previous_state])
    #         # Remember the probabilities from the previous layer
    #         previous_layer_dict = deepcopy(current_layer_dict)
    #         previous_states = deepcopy(current_states_temp)
    #         out.append(current_layer_dict)
    #
    #     return out

    def sample_forward(self, adjacency_matrix, n_samples=None, timer=False, saveFilename=None):
        """
        Given an adjacency matrix, cluster some graphs and return
        'epochs' number of final states reached using the current
        forward model. If epochs is left as None use self.epochs.
        :param adjacency_matrix: matrix (n_nodes, n_nodes)
        :param n_samples: (None or int)
        :return:
        """

        if n_samples is None:
            n_samples = self.epochs

        final_states = torch.zeros((n_samples, self.state_length))
        for epoch in tqdm(range(n_samples), desc='Sampling') if timer else range(n_samples):
            # Initialize the empty clustering and one-hot vector
            clustering_matrix = torch.zeros(self.size)
            clustering_list = torch.zeros(self.n_nodes)
            current_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))
            start = True
            # Each iteration has self.termination_chance of being the last, and we ensure no empty states
            while start or (
                    torch.rand(1) >= self.termination_chance and torch.sum(clustering_list > 0) != self.n_nodes):
                start = False
                # Pass the state vector to the NN
                clustering_list, num_clusters = self.get_clustering_list(clustering_matrix)
                forward_flows = self.forward_flow(current_state).flatten()
                max_val = torch.max(forward_flows)
                forward_probs = self.softmax_matrix(forward_flows - max_val)
                # Sample from the output and retrieve the indices of the chosen node and clustering
                index_chosen = torch.multinomial(forward_probs, 1)
                # First find the row and column we have chosen from the probs
                node_chosen = index_chosen // num_clusters
                cluster_index_chosen = index_chosen % num_clusters
                # Next, increment the node chosen by the number of nodes ahead of it that were already in the graph
                indicies_available = torch.argwhere(clustering_list == 0)
                node_chosen = indicies_available[node_chosen][0]  # Argwhere produces a 2 dimensional array
                # Update the cluster
                # Labels are 1-indexed, indices are 0 indexed
                clustering_list[node_chosen] = torch.tensor(cluster_index_chosen + 1, dtype=torch.float32)
                clustering_matrix = self.get_clustering_matrix(clustering_list, num_clusters)
                current_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))
            # Catch empty clusterings to see why they occur
            assert sum(clustering_list) >= self.n_nodes
            final_states[epoch] = current_state

        if saveFilename is not None:
            print('SAVING STATES!')
            torch.save(final_states, saveFilename + '.pt')
        return final_states

    def full_sample_distribution_G(self, adjacency_matrix, log=True, fix=False):
        """
        Computes the exact forward sample probabilities
        for each possible clustering. Calculates each step
        from the previous and thus returns a list of dictionaries
        where each entry corresponds to a layer, starting from
        an empty clustering. If fix is True, it returns the
        posterior values in order of the sorting, along with the
        dictionary.
        :param log: (bool) Whether or not to compute the function using log-probabilities. This doesn't work, as we have to sum probabilities for multiple avenues.
        :return: list of dictionaries of the form {clustering_matrix: probability}, s
        """
        print(
            "Warning: You are embarking on the long and arduous journey of calculating all the forward sample probabilities exactly. This scales poorly with N and might take a while.")
        # from copy import deepcopy

        # Initialize the empty clustering and one-hot vector (source state)
        clustering_matrix = torch.zeros(self.size)
        clustering_list = torch.zeros(self.n_nodes)
        next_states_p = [defaultdict(lambda: 0) for _ in
                         range(self.n_nodes + 1)]  # Must be overwritten in the case of log, as log(0) isn't defined.
        state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))  # Init_state
        prob_init = 0 if log else 1
        next_states_p[0] = {clustering_list: prob_init}  # Transition to state 0
        for n_layer in tqdm(range(0, self.n_nodes), desc='GFlowNet Output'):
            for clustering_list, prob in next_states_p[n_layer].items():
                num_clusters = len(torch.unique(clustering_list)) - 1
                clustering_matrix = self.get_clustering_matrix(clustering_list, num_clusters)
                state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))

                output = self.forward_flow(state)
                # output = torch.zeros((nodes_to_place.size()[0], number_of_clusters))
                # shape = (nodes left, number of clusters (+1))
                Fabian = True  # Er det softmax eller er det ikke.
                if not Fabian:
                    if not log:
                        output_prob = (output / torch.sum(output))
                    else:
                        output_prob = (torch.log(output) - torch.log(torch.sum(output)))
                else:
                    if not log:
                        output_prob = self.softmax_matrix(output)
                    else:
                        output_prob = torch.log(self.softmax_matrix(output))

                for index_chosen, next_prob in enumerate(output_prob.flatten()):
                    new_state, temp_clustering_list = self.place_node(state, index_chosen,
                                                                      return_clustering_list=True)  # This ends up representing identical clusterings differently. We fix that.
                    temp_num_clusters = max(num_clusters, 1 + (index_chosen % (
                            num_clusters + 1)))  # Just a clever way of figuring out how many clusters there are because I am being cheeky.
                    temp_clustering_matrix = self.get_clustering_matrix(temp_clustering_list,
                                                                        temp_num_clusters)  # We could rewrite this function to not need the number of clusters
                    temp_clustering_list = self.get_clustering_list(temp_clustering_matrix)[0]
                    # Use the clustering list as the keys in the dictionary to save space
                    if not log:
                        next_states_p[n_layer + 1][temp_clustering_list] += (prob * next_prob)
                    else:
                        if next_states_p[n_layer + 1].get(temp_clustering_list, 0) == 0:
                            next_states_p[n_layer + 1][
                                temp_clustering_list] = prob + next_prob  # Initialize the value.
                        else:
                            # There are some funny issues here with not being able to use tensors as keys.
                            next_states_p[n_layer + 1][temp_clustering_list] = torch.logaddexp(
                                next_states_p[n_layer + 1][temp_clustering_list], prob + next_prob)
            assert -0.1 < torch.logsumexp(torch.tensor(list(next_states_p[n_layer + 1].values())), (0)) < 0.1

        if not fix:
            return next_states_p
        else:
            return next_states_p, self.fix_net_clusters(next_states_p,
                                                        log=log)  # I have no idea why the earlier turnary didn't work, but it didn't

    def fix_net_clusters(self, cluster_prob_dict, log=True):
        clusters_all = allPermutations(self.n_nodes)
        Bell, N = clusters_all.shape
        net_posteriors = torch.zeros(Bell)
        clusters_all_tensor = torch.tensor(clusters_all + 1)
        assert -0.1 < torch.logsumexp(torch.tensor(list(cluster_prob_dict[N].values())),
                                      (0)) < 0.1  # Make sure that the probabilities sum to 1.
        for net_c, post in cluster_prob_dict[N].items():
            # Vectorize this because I can.
            cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor, net_c), dim=1) == 1)[0][0]
            if not log:
                net_posteriors[cluster_ind] += post
            else:
                if net_posteriors[cluster_ind] == 0:
                    net_posteriors[cluster_ind] = post
                else:
                    net_posteriors[cluster_ind] = torch.logaddexp(net_posteriors[cluster_ind], post)
        assert -0.1 < torch.logsumexp(net_posteriors, (0)) < 0.1
        return net_posteriors

    def plot_full_distribution(self, adjacency_matrix, title='', filename_save='', log=True):
        """
        Plots the entire distribution of IRM values along
        with the NN's output for visual comparison. Given
        the adjacency matrix. The plot can be saved by
        specifying a filename, the title can be specified
        and 'log' determines whether to keep the values in
        the log domain.
        :param adjacency_matrix:
        :param title:
        :param filename_save:
        :param log:
        :return:
        """

        clusters_all = allPermutations(self.n_nodes)
        Bell = len(clusters_all)
        clusters_all_post = np.zeros(Bell)

        for i, cluster in enumerate(clusters_all):
            posterior = (torch_posterior(adjacency_matrix, cluster, a=torch.tensor(self.a), b=torch.tensor(self.b),
                                         A_alpha=torch.tensor(self.A_alpha), log=True))
            clusters_all_post[i] = posterior

        cluster_post = clusters_all_post - logsumexp(clusters_all_post)

        # Normalize them into proper log probabilities
        if not log: cluster_post = np.exp(cluster_post)
        sort_idx = np.argsort(cluster_post)

        cluster_prob_dict, fixed_probs = self.full_sample_distribution_G(adjacency_matrix=adjacency_matrix, log=True,
                                                                         fix=True)
        # net_probs = [float(value) for key, value in fixed_probs]

        f = plt.figure()
        values_real = cluster_post[sort_idx]
        values_network = fixed_probs.detach().numpy()[sort_idx]
        plt.plot(values_real, "o", label='IRM Values')
        plt.plot(values_network, "o", label='GFlowNet values')

        plt.title(title) if title else plt.title('Cluster Posterior Probabilites')
        plt.xlabel("Cluster Index")
        plt.ylabel("Posterior Probability")
        plt.xticks(np.arange(1, len(fixed_probs) + 1))
        plt.legend()
        if filename_save:
            plt.savefig(filename_save + '.png')
        plt.show()

        return cluster_post, fixed_probs, sort_idx

    def predict(self, x):
        """
        Given a state, return the forward model's
        prediction
        :param x:
        :return:
        """
        return self.model_forward.forward(x)

    # %% Helpers:
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
        number_of_clusters = 1  # starting at the empty cluster
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
        :return: adjacency matrix and clustering matrix (n_nodes, n_nodes)
        """
        # returns the adjacency matrix, clustering matrix
        l = self.n_nodes ** 2
        size = self.size
        current_adjacency, current_clustering = torch.reshape(state[:l], size), torch.reshape(state[l:2 * l], size)
        return current_adjacency, current_clustering

    def place_node(self, state, index_chosen: int, return_clustering_list=False):
        """
        Given a state and the index we happen to choose as
        an integer, from the flattened output array, return
        the next state with the chosen node in the chosen
        cluster.
        :param state:
        :param index_chosen:
        :param return_clustering_list:
        :return: new_state:
        """
        adjacency_matrix, clustering_matrix = self.get_matrices_from_state(state)
        clustering_list, num_clusters = self.get_clustering_list(clustering_matrix)

        node_chosen = index_chosen // num_clusters
        cluster_index_chosen = index_chosen % num_clusters
        # Next, increment the node chosen by the number of nodes ahead of it that were already in the graph
        indicies_available = torch.argwhere(clustering_list == 0)
        node_chosen = indicies_available[node_chosen][0]  # Argwhere produces a 2 dimensional array
        # Update the cluster
        # Labels are 1-indexed, indices are 0 indexed
        clustering_list[node_chosen] = torch.tensor(cluster_index_chosen + 1, dtype=torch.float32)
        clustering_matrix = self.get_clustering_matrix(clustering_list, num_clusters)
        new_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))
        if return_clustering_list:
            return new_state, clustering_list
        return new_state

    def softmax_matrix(self, inp):
        """
        Takes the softmax across an entire matrix/vector of *any* size
        :param inp:
        :return:
        """
        return torch.exp(inp) / torch.sum(torch.exp(inp))

    def is_terminal(self, state):
        """
        Given a state vector, check if it is in a terminal state
        by summing the diagonal of the clustering matrix and
        checking if it is equal to the number of nodes
        :param state:
        :return:
        """
        # Check the diagonal of the clustering matrix using indexing and if the sum == n_nodes it is terminal
        _, clustering_matrix = self.get_matrices_from_state(state)
        return torch.diag(clustering_matrix).sum() == self.n_nodes

    def __str__(self):
        return f'GFlowNet Object with {self.n_nodes} nodes'

    def save(self, prefix='GFlowNet', postfix=''):
        torch.save(self.model_forward.state_dict(), prefix + 'Forward' + postfix + '.pt')
        # torch.save(self.model_backward.state_dict(), prefix + 'Backward' + postfix + '.pt')

    def load_forward(self, prefix='GFlowNet', postfix=''):
        self.model_forward.load_state_dict(torch.load(prefix + 'Forward' + postfix + '.pt'))

    def load_backward(self, prefix='GFlowNet', postfix=''):
        self.model_backward.load_state_dict(torch.load(prefix + 'Backward' + postfix + '.pt'))

    def load_samples(self, filename):
        return torch.load(filename)


class GraphNetNodeOrder(GraphNet):
    def __init__(self, n_nodes, **kwargs):
        super().__init__(n_nodes, **kwargs)

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
        adjacency_matrix, clustering_matrix = self.get_matrices_from_state(terminal_state)
        # To prevent changes to the original values??
        adjacency_matrix, clustering_matrix = adjacency_matrix.clone(), clustering_matrix.clone()
        current_clustering_list, number_of_clusters = self.get_clustering_list(clustering_matrix)
        prob_log_sum_forward = torch.zeros(1)
        # Need IRM to add here, leave out for now
        prob_log_sum_backward = torch.zeros(1)
        # Will track the order in which nodes are removed from the graph
        trajectory = torch.zeros(self.n_nodes)
        node_no = 1
        current_state = terminal_state

        while torch.sum(current_clustering_list).item():
            probs_backward = self.model_backward.forward(current_state) if self.using_backward_model \
                else self.model_backward.forward(clustering_matrix)
            index_chosen = torch.multinomial(probs_backward, 1)
            prob_log_sum_backward += torch.log(probs_backward[index_chosen])
            # Remove the node from the clustering
            clustering_matrix[index_chosen] = 0
            clustering_matrix[:, index_chosen] = 0
            # Get the forward flow
            current_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))
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

    def forward_flow(self, state, node_index=0):
        """
        Given a state, return the forward flow from the current
        forward model.
        :param state: vector ((n_nodes^2) * 2,)
        :return:
        """
        # return the forward flow for all possible actions (choosing node, assigning cluster)
        # using the idea that the NN rates each possible future state. Variable output size!!
        current_adjacency, current_clustering = self.get_matrices_from_state(state)
        # number of clusters starts at 1
        clustering_list, number_of_clusters = self.get_clustering_list(current_clustering)

        output = torch.zeros(number_of_clusters)

        for possible_cluster in range(1, number_of_clusters + 1):
            temp_clustering_list = torch.clone(clustering_list)
            temp_clustering_list[node_index] = possible_cluster

            temp_clustering = self.get_clustering_matrix(temp_clustering_list, number_of_clusters)
            temp_state = torch.concat((current_adjacency.flatten(), temp_clustering.flatten()))
            output[possible_cluster - 1] = self.model_forward.forward(temp_state)
        # assert not any((torch.isinf(output).flatten() + torch.isnan(output).flatten()))
        return output

    def sample_forward(self, adjacency_matrix, n_samples=None, timer=False, saveFilename=None):
        """
        Given an adjacency matrix, cluster some graphs and return
        'epochs' number of final states reached using the current
        forward model. If epochs is left as None use self.epochs.
        :param adjacency_matrix: matrix (n_nodes, n_nodes)
        :param n_samples: (None or int)
        :return:
        """

        if n_samples is None:
            n_samples = self.epochs

        final_states = torch.zeros((n_samples, self.state_length))
        for epoch in tqdm(range(n_samples), desc='Sampling') if timer else range(n_samples):
            # Initialize the empty clustering and one-hot vector
            clustering_matrix = torch.zeros(self.size)
            clustering_list = torch.zeros(self.n_nodes)
            current_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))
            # Each iteration has self.termination_chance of being the last, and we ensure no empty states
            for node_index in np.random.permutation(self.n_nodes):
                # Pass the state vector to the NN
                clustering_list, num_clusters = self.get_clustering_list(clustering_matrix)
                forward_flows = self.forward_flow(current_state, node_index=node_index).flatten()
                max_val = torch.max(forward_flows)
                forward_probs = self.softmax_matrix(forward_flows - max_val)
                # Sample from the output and retrieve the indices of the chosen node and clustering
                index_chosen = torch.multinomial(forward_probs, 1)
                # First find the row and column we have chosen from the probs
                cluster_index_chosen = index_chosen % num_clusters
                # Next, increment the node chosen by the number of nodes ahead of it that were already in the graph
                # Labels are 1-indexed, indices are 0 indexed
                clustering_list[node_index] = torch.tensor(cluster_index_chosen + 1, dtype=torch.float32)
                clustering_matrix = self.get_clustering_matrix(clustering_list, num_clusters)
                current_state = torch.concat((adjacency_matrix.flatten(), clustering_matrix.flatten()))
            # Catch empty clusterings to see why they occur
            assert sum(clustering_list) >= self.n_nodes
            final_states[epoch] = current_state

        if saveFilename is not None:
            torch.save(final_states, saveFilename + '.pt')
        return final_states


# %% NN
class MLP(nn.Module):
    def __init__(self, n_hidden, n_nodes, n_clusters, output_size=1, n_layers=3, softmax=False):
        super().__init__()
        # takes adj_mat, clustering_mat, one-hot node_placed
        input_size = int(2 * n_nodes ** 2)
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
        # To ensure positive output (DO NOT ENSURE THIS!!!)
        if softmax:
            self.layers.append(nn.Softmax(dim=0))

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


# %% Graph Theory functions
def p_x_giv_z(A, C, a=1, b=1, log=True):
    """Calculate P(X|z): the probability of the graph given a particular clustering structure.
    # This is calculated by integrating out all the internal cluster connection parameters.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (1D ndarray) (n long with the cluster c of each node ordered by the Adjacency matrix at each index)
    a and b: float
        Parameters for the beta distribution prior for the cluster connectivities.
        a = b = 1 yields a uniform distribution.
    log : Bool
        Whether or not to return log of the probability

    Return
    ----------
    Probability of data given clustering: float
    """
    np.einsum("ii->i", A)[
        ...] = 0  # This function assumes that nodes aren't connected to themselves. This should be irrelevant for the clustering.
    # Product over all pairs of components.
    values, nk = np.unique(C, return_counts=True)
    # I just need to create m_kl and m_bar_kl matrices. Then I can vectorize the whole thing

    # create node-cluster adjacency matrix
    n_C = np.identity(C.max() + 1, int)[C]
    m_kl = n_C.T @ A @ n_C

    np.einsum("ii->i", m_kl)[
        ...] //= 2  # np.diag(m_kl)[...] //= 2 , but this allows for an in-place update.
    m_bar_kl = np.outer(nk, nk) - np.diag(nk * (nk + 1) / 2) - m_kl  # The diagonal simplifies to the sum up to nk-1.

    # Alternative to the matrix multiplication. This is a little faster.
    # Sort A according to the clustering.
    # boundaries = np.diff(C, prepend=-1).nonzero()[0] # tells me at what index each cluster begins.
    # out =  np.add.reduceat(np.add.reduceat(A,boundaries,1),boundaries,0) # Basically just the matrix-algebra above.

    logP_x_giv_z = np.sum(np.triu(betaln(m_kl + a, m_bar_kl + b) - betaln(a, b)))

    return logP_x_giv_z if log else np.exp(logP_x_giv_z)


def p_z(A, C, A_alpha=1, log=True):
    """Probability of clustering.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (ndarray)
    A_alpha : float
        Total concentration of clusters.
    log : Bool
        Whether or not to return log of the probability

    Return
    ----------
    probability of cluster: float
    """

    # A_alpha is the total concentration parameter.
    # A constant concentration corrosponds to the chinese restaurant process. 
    N = len(A)
    values, nk = np.unique(C,
                           return_counts=True)  # nk is an array of counts, so the number of elements in each cluster.
    K_bar = len(values)  # number of non empty clusters.

    # nk (array of number of nodes in each cluster)
    log_p_z = (gammaln(A_alpha) + K_bar * (np.log(A_alpha)) - gammaln(A_alpha + N)) + np.sum(gammaln(nk))

    return log_p_z if log else np.exp(log_p_z)


def torch_posterior(A_in, C_in, a=None, b=None, A_alpha=None, log=True, verbose=False):
    """Calculate P(X,z): the joint probability of the graph and a particular clustering structure. This is proportional to the posterior.
    # This is calculated by integrating out all the internal cluster connection parameters.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (1D ndarray) (n long with the cluster c of each node ordered by the Adjacency matrix at each index)
        Importantly it is 0 indexed! (Write to me if I should change this. We should be consistent on this, and it is probably best to keep 0 as unclustered)
    a and b: float
        Parameters for the beta distribution prior for the cluster connectivities.
        a = b = 1 yields a uniform distribution.
    A_alpha : float
        Total concentration of clusters.
    log : Bool
        Whether or not to return log of the probability
    verbose: Bool
        Whether or not to return the part of the computer this computation is computed on.

    Return
    ----------
    Probability of data and clustering: float
    """
    assert 0 in C_in  # All nodes should be clustered and the clusters should be 0-indexed. 0 must be in C_in. # We really should decide on one standard here.

    # Likelihood part
    if a is None:
        a = torch.ones(1)
    if b is None:
        b = torch.ones(1)
    if A_alpha is None:
        A_alpha = torch.ones(1)

    A = torch.t_copy(A_in)
    C = torch.t_copy(torch.tensor(C_in, dtype=torch.int64))
    torch.einsum("ii->i", A)[...] = 0  # Fills the diagonal with zeros.
    values, nk = torch.unique(C, return_counts=True)
    n_C = torch.eye(int(C.max()) + 1)[C]

    m_kl = n_C.T @ A @ n_C
    torch.einsum("ii->i", m_kl)[
        ...] /= 2  # m_kl[np.diag_indices_form(m_kl)] //= 2 should do the same thing. Will always be an integer.

    m_bar_kl = torch.outer(nk, nk) - torch.diag(nk * (nk + 1) / 2) - m_kl

    if verbose: print(m_bar_kl.device, m_kl.device)

    if str(a.device)[:4] == 'cuda':
        a_ = a.detach().cpu().numpy()
        b_ = b.detach().cpu().numpy()
        m_kl_ = m_kl.detach().cpu().numpy()
        m_bar_kl_ = m_bar_kl.detach().cpu().numpy()
        logP_x_giv_z = torch.tensor(np.sum(torch.triu(betaln(m_kl_ + a_, m_bar_kl_ + b_) - betaln(a_, b_))))
    else:
        logP_x_giv_z = torch.sum(torch.triu(betaln(m_kl + a, m_bar_kl + b) - betaln(a, b)))

    # Prior part. P(z|K), så given K possible labellings.
    N = len(A)
    # values, nk = torch.unique(C, return_counts=True)
    K_bar = len(values)  # number of non empty clusters.

    log_p_z = (torch_gammaln(A_alpha) + K_bar * (torch.log(A_alpha)) - torch_gammaln(A_alpha + N)) + torch.sum(
        torch_gammaln(nk))
    # Return joint probability, which is proportional to the posterior
    return logP_x_giv_z + log_p_z if log else torch.exp(logP_x_giv_z + log_p_z)


def Cmatrix_to_array(Cmat):
    C = np.zeros(len(Cmat))
    cluster = 0
    for i, row in enumerate(Cmat):  # row = Cmat[i]
        if np.any(Cmat[i]):
            C[Cmat[i].astype(bool)] = cluster
            Cmat[Cmat[i].astype(bool)] = 0  # Remove these clusters
            cluster += 1
    return C


def ClusterGraph(l, k, p, q):
    n = l * k
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(n):

            prob = np.random.rand(2)

            if i // k == j // k and prob[0] < p:
                adjacency[(i, j), (j, i)] = 1

            elif prob[1] < q:
                adjacency[(i, j), (j, i)] = 1

    return adjacency


# Collected function
def IRM_graph(A_alpha, a, b, N):
    clusters = CRP(A_alpha, N)
    phis = Phi(clusters, a, b)
    Adj = Adj_matrix(phis, clusters)
    return Adj, clusters


# Perform Chinese Restaurant Process
def CRP(A_alpha, N):
    # First seating
    clusters = [[1]]
    for i in range(2, N + 1):
        # Calculate cluster assignment as index to the list clusters.
        p = torch.rand(1)
        probs = torch.tensor([len(cluster) / (i - 1 + A_alpha) for cluster in clusters])
        cluster_assignment = sum(torch.cumsum(probs, dim=0) < p)

        # Make new table or assign to current
        if cluster_assignment == len(clusters):
            clusters.append([i])
        else:
            clusters[cluster_assignment].append(i)

    # Return the cluster sizes
    return torch.tensor([len(cluster) for cluster in clusters])


# Return a symmetric matrix of cluster probabilities,
# defined by a beta distribution.
def Phi(clusters, a, b):
    n = len(clusters)
    phis = Beta(a, b).rsample((n, n))
    # Symmetrize
    for i in range(n - 1, -1, -1):
        for j in range(n):
            phis[i, j] = phis[j, i]

    return phis


# Helper function to construct block matrix of cluster probabilities.
def make_block_phis(phis, clusters):
    for i, ii in enumerate(clusters):
        for j, jj in enumerate(clusters):
            if j == 0:
                A = torch.full((ii, jj), phis[i, j])
            else:
                A = torch.hstack((A, torch.full((ii, jj), phis[i, j])))

        if i == 0:
            block_phis = A
        else:
            block_phis = torch.vstack((block_phis, A))

    return block_phis


# Construct adjacency matrix.
def Adj_matrix(phis, clusters):
    n = sum(clusters)
    Adj_matrix = torch.zeros((n, n))

    block_phis = make_block_phis(phis, clusters)

    # Iterate over all nodes and cluster probabilities.
    for i in range(n):
        for j in range(n):
            p = torch.rand(1)
            if p < block_phis[i, j]:
                Adj_matrix[i, j] = 1
                Adj_matrix[j, i] = 1
            else:
                Adj_matrix[i, j] = 0
                Adj_matrix[j, i] = 0

    return Adj_matrix


def clusterIndex(clusters):
    idxs = torch.tensor([])
    for i, k in enumerate(clusters):
        idxs = torch.cat((idxs, torch.tensor([i] * k)))
    return idxs


def allPermutations(n):
    """
    Return a list of all possible permutations of clustering
    lists for a graph with n nodes
    :param n: int
    :return: numpy array
    """
    perm = [[[1]]]
    for i in range(n - 1):
        perm.append([])
        for partial in perm[i]:
            for j in range(1, max(partial) + 2):
                perm[i + 1].append(partial + [j])

    return np.array(perm[-1]) - 1


def allPosteriors(A_random, a, b, A_alpha, log, joint = False):
    # Computing posteriors for all clusters.
    N = len(A_random)
    clusters_all = allPermutations(N)
    Bell = len(clusters_all)
    clusters_all_post = np.zeros(Bell)
    for i, cluster in enumerate(clusters_all):
        posterior = torch_posterior(A_random, cluster, a=torch.tensor(a), b=torch.tensor(b), A_alpha = torch.tensor(A_alpha), log= True)
        clusters_all_post[i] = posterior
    if joint: return clusters_all_post # Return the joint probability instead of normalizing.
    cluster_post = clusters_all_post - logsumexp(clusters_all_post) # Normalize them into proper log probabilities
    if not log: cluster_post = np.exp(cluster_post)
    return cluster_post


# %% EXTRAS
def print_Latex_table(table, significantFigures=3, headerRow=None, indexColumn=None):
    """
    Given a table (2D array), print the values rounded to
    a specified number of significant figures in a format
    one can cut and paste directly into a LateX table.
    headerRow be an iterable of any stringable type, while
    indexColumn must be an iterable of strings containing
    the index and subsequent ' & '. The title of the indexes
    is not printed, and must be added manually.
    :param table:
    :param significantFigures:
    :return:
    """
    if indexColumn is None:
        indexColumn = ['' for _ in table[:, 0]]
    print('{' + '|c' * table.shape[1] + '|}')
    if headerRow is not None:
        print('\\hline ')
        row_str = ''
        for el in headerRow:
            row_str += f'{el} & '
        row_str = row_str[:-2] + '\\\\'
        print(row_str)

    for index, row in zip(indexColumn, signif(table, significantFigures)):
        print('\\hline ')
        row_str = index
        for el in row[:-1]:
            row_str += f'{el} & '
        row_str = row_str[:-2] + '\\\\'
        print(row_str)
    print('\\hline ')


def signif(x, p):
    """
    Round an array x to p significant figures
    :param x:
    :param p:
    :return:
    """
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def check_gpu():
    print(f'Cuda is {"available" if torch.cuda.is_available() else "not available"}')


def compare_results_small_graphs(filename,
                                 min_N=3,
                                 max_N=4,
                                 max_epochs=100,
                                 epoch_interval=10,
                                 n_samples=100,
                                 n_samples_distribution=2000,
                                 run_test=False,
                                 using_backward_model=False,
                                 train_mixed=False,
                                 use_fixed_node_order=False,
                                 plot_last=False,
                                 a=1.,
                                 b=1.,
                                 A_alpha=1.):
    """
    Given a destination file, calculate and store the difference
    between the GFlowNet's output and the true IRM values,
    trained on graphs of increasing N for an increasing number
    of epochs.
    :param filename:
    :param min_N:
    :param max_N:
    :param max_epochs:
    :param epoch_interval:
    :param using_backward_model:
    :return:
    """
    with open(filename, 'w') as file:
        # Create the first line, rows are nodes, columns are epochs
        file.write('N,')
        for epochs in range(0, max_epochs + 1, epoch_interval):
            file.write(f'{epochs},')
        file.write('\n')
        fully_trained_networks = []
        for N in tqdm(range(min_N, max_N + 1), desc='Iterating over N'):
            file.write(f'{N},')

            adjacency_matrix, clusters = IRM_graph(A_alpha=A_alpha, a=a, b=b, N=N)
            while sum(adjacency_matrix.flatten()) == 0:  # No unconnected graphs
                adjacency_matrix, clusters = IRM_graph(A_alpha=A_alpha, a=a, b=b, N=N)

            # Use the same net object, just tested every epoch_interval
            net = GraphNet(n_nodes=adjacency_matrix.size()[0], a=a, b=b, A_alpha=A_alpha,
                           using_backward_model=using_backward_model) if not use_fixed_node_order else GraphNetNodeOrder(
                n_nodes=adjacency_matrix.size()[0], a=a, b=b, A_alpha=A_alpha,
                using_backward_model=using_backward_model)

            # Train using the sampled values before any training
            # if train_mixed:
            #     X = torch.zeros((n_samples, net.n_nodes ** 2 * 2))
            #     for i in range(n_samples):
            #         X[i] =

            # X = net.sample_forward(adjacency_matrix, n_samples=n_samples)

            cluster_post = allPosteriors(adjacency_matrix, a, b, A_alpha, log=True, joint=False)
            # cluster_prob_dict = net.full_sample_distribution_G(adjacency_matrix=adjacency_matrix,
            #                                                    log=True,
            #                                                    fix=False)
            X1 = net.sample_forward(adjacency_matrix, n_samples=n_samples_distribution)
            sample_posteriors_numpy = empiricalSampleDistribution(X1, N, net, log=True, numpy=True)
            sort_idx = np.argsort(cluster_post)
            difference = sum(abs(cluster_post[sort_idx] - sample_posteriors_numpy[sort_idx]))
            file.write(f'{difference},')
            for epochs in range(0, max_epochs + epoch_interval, epoch_interval):
                losses = net.train(X1, epochs=epoch_interval * (epochs != 0), verbose=True)
                # cluster_prob_dict = net.full_sample_distribution_G(adjacency_matrix=adjacency_matrix,
                #                                                                 log=True,
                #                                                                 fix=False)
                # fixed_probs = net.fix_net_clusters(cluster_prob_dict, log=True)
                X1 = net.sample_forward(adjacency_matrix, n_samples=n_samples_distribution, timer=True)
                sample_posteriors_numpy = empiricalSampleDistribution(X1, N, net, log=True, numpy=True)
                sort_idx = np.argsort(cluster_post)
                difference = sum(abs(cluster_post[sort_idx] - sample_posteriors_numpy[sort_idx]))
                file.write(f'{difference},')
            file.write('\n')
            fully_trained_networks.append(net)
            if plot_last:
                plot_posterior(cluster_post,
                               sort_idx=sort_idx,
                               net_posteriors_numpy=None,
                               sample_posteriors_numpy=sample_posteriors_numpy,
                               saveFilename=f'Plots/PosteriorPlot_{N}_{max_epochs}_{n_samples_distribution}.png')

        if run_test:
            test_results = []
            for network in tqdm(fully_trained_networks):
                N = network.n_nodes
                test_temp = []
                adjacency_matrix_test, clusters_test = IRM_graph(A_alpha=A_alpha, a=a, b=b, N=N)

                cluster_post = allPosteriors(adjacency_matrix_test, a, b, A_alpha, log=True, joint=False)
                X = network.sample_forward(adjacency_matrix_test, n_samples=n_samples)

                for epochs in range(0, max_epochs + 1, epoch_interval):
                    losses = network.train(X, epochs=epoch_interval, verbose=True)
                    cluster_prob_dict, fixed_probs = network.full_sample_distribution_G(
                        adjacency_matrix=adjacency_matrix_test,
                        log=True,
                        fix=True)
                    difference = sum(abs(cluster_post - fixed_probs.detach().numpy()))
                    test_temp.append(difference)
                test_results.append(test_temp)
            pd.DataFrame(test_results).to_csv(filename[:-4] + '_TEST.csv')
    return fully_trained_networks


def plot_posterior(cluster_post, sort_idx = None, net_posteriors_numpy = None, sample_posteriors_numpy = None, log = True, saveFilename=''):
    log_string = " Log " if log else " "
    order = "index" if (sort_idx is None) else "Magnitude"
    xlab = "Cluster Index" if (sort_idx is None) else "Sorted Cluster Index"
    if sort_idx is None: sort_idx = torch.arange(len(cluster_post))

    from_network = '' if ((net_posteriors_numpy is None) or (sample_posteriors_numpy is None)) else ':\nExact and extracted from network'
    f = plt.figure()
    plt.title('Cluster Posterior' + log_string + 'Probabilites by ' + order + from_network)
    if not log: cluster_post = np.exp(cluster_post)
    plt.plot(cluster_post[sort_idx], "bo")
    if net_posteriors_numpy is not None:
        if not log: net_posteriors_numpy = np.exp(net_posteriors_numpy)
        plt.plot(net_posteriors_numpy[sort_idx], "ro", markersize=4)
    if sample_posteriors_numpy is not None:
        if not log: sample_posteriors_numpy = np.exp(sample_posteriors_numpy)
        plt.plot(sample_posteriors_numpy[sort_idx], "gx")
    plt.xlabel(xlab)
    plt.ylabel("Posterior Probability")
    plt.legend(["Exact values", "From Network", "Sampled Empirically"])
    # plt.ylim(0, -5)
    plt.tight_layout()
    if saveFilename:
        plt.savefig(saveFilename)
    plt.show()
    return

# %% Gibbs Sampler
def Gibbs_likelihood(A, C, a=0.5, b=0.5, log=True):
    # WRONG!!!
    """Calculate Gibbs_likelyhood as presented in Mikkel's paper.

    Parameters
    ----------
    A : Adjacency matrix (2D ndarray)
    C : clustering index array (1D ndarray) (number clustered nodes long with the cluster c of each node ordered by the Adjacency matrix at each index)
            (0 Indexed. 1 Indexed should work as well, since it adds an empty initial cluster which doesn't change the likelyhood, but I am not sure.)
    a and b: float
        Parameters for the beta distribution prior for the cluster connectivities.
        a = b = 1 yields a uniform distribution.
    log : Bool
        Whether or not to return log of the probability

    Return
    ----------
    Array of Gibbs_likelyhoods for each cluster k and +1 for a new cluster: float
    """

    # Here lies the old incorrect version as a note.
    if C.size == 0: return (np.zeros(1) if log else np.ones(1))

    C = C.astype(int)
    values, nk = np.unique(C, return_counts=True)
    A = A[:(len(C) + 1), :(len(C) + 1)]
    np.einsum("ii->i", A)[...] = 0  # Beware. This is wrong.

    n_C = np.identity(C.max() + 1, int)[C]  # create node-cluster adjacency matrix
    n_C = np.vstack((n_C, np.zeros(C.max() + 1, int)))  # add an empty last row as a place holder for the last node.
    n_C = np.hstack((n_C, np.zeros((len(C) + 1, 1),
                                   int)))  # add an empty last partition with no nodes in it for the new cluster option.
    nk = np.append(nk, 0)  # The last extra cluster has no nodes.
    r_nl_matrix = (A @ n_C)  # node i connections to each cluster.
    r_nl = r_nl_matrix[len(C)]  # just node n. (Array)

    m_kl = n_C.T @ A @ n_C
    np.einsum("ii->i", m_kl)[...] //= 2
    m_bar_kl = np.outer(nk, nk) - np.diag(nk * (nk + 1) / 2) - m_kl

    Gibbs_log_likelyhood = np.sum(betaln(m_kl + r_nl + a, m_bar_kl + nk - r_nl + b) - betaln(m_kl + a, m_bar_kl + b),
                                  axis=1).reshape((-1))

    return Gibbs_log_likelyhood if log else np.exp(Gibbs_log_likelyhood)


def Gibbs_sample_np(A, T, burn_in_buffer=None, sample_interval=None, seed=42, a=1, b=1, A_alpha=1,
                    return_clustering_matrix=False):
    """Perform one Gibbs sweep with a specified node order to yield one sampled clustering.

    Parameters
    ----------
    A : (2D ndarray) Adjacency matrix.
    T : (int) Number of Gibbs Sweeps.
    burn_in_buffer : (int) Number of initial samples to discard. Default is T//2.
    sample_interval : (int) The sweep interval with which to save clusterings.
    seed : (int) Seed used for random node permutations.
    a, b and A_alpha: float
        Parameters for the beta distribution prior for the cluster connectivities and concentration.
        a = b = 1 yields a uniform distribution.
        A_alpha is the total cluster concentration parameter.
    return_clustering_matrix: (bool) a list of full clustering adjacency matrices instead of partial one-hot encoded clustering matrices.

    Return
    ----------
    Z: (list) list of clusterings z sampled through iterations of Gibbs sweeps.
    """
    assert isinstance(A, np.ndarray)
    A[np.diag_indices_from(
        A)] = 0  # Assume nodes aren't connected to themselves. Equivalent to np.einsum("ii->i", A)[...] = 0

    N = A.shape[0]
    # z_f = np.zeros((N,N)) # z_full. Take all the memory I need, which is not that much.
    # z_f[:, 0] = 1 # Initialize everthing to be in the first cluster.
    z = np.ones((N, 1))
    np.random.seed(seed=seed)  # torch.manual_seed(seed) torch.cuda.manual_seed(seed)
    Z = []
    for t in range(T):
        node_order = np.random.permutation(
            N)  # I could also make the full (N,T) permutations matrix to exchange speed for memory
        for i, n in enumerate(node_order):
            # nn = np.delete(node_order, i, axis = 0)
            nn_ = np.arange(N)  # Another option that is a bit simpler and doesn't permute quite as hard.
            nn = nn_[nn_ != n]
            m = np.sum(z[nn, :], axis=0)  # Could also use my old cluster representation, which is probably faster.
            K_ind = np.nonzero(m)[0]
            K = len(K_ind)  # Number of clusters
            z, m = z[:, K_ind], m[K_ind]  # Fix potential empty clusters

            m_kl = z[nn, :].T @ A[nn][:, nn] @ z[nn, :]
            m_kl[np.diag_indices_from(
                m_kl)] //= 2  # The edges are undirected and the previous line counts edges within a cluster twice as if directed. Compensate for that here.

            m_bar_kl = np.outer(m, m) - np.diag(m * (m + 1) / 2) - m_kl
            r_nl = (z[nn, :].T @ A[nn, n])  # node n connections to each cluster (l).

            n_C = z
            r_nl_matrix = (A @ n_C)
            assert np.all(r_nl == r_nl_matrix[n])

            # Calculate the big log posterior for the cluster allocation for node n. (k+1 possible cluster allocations)
            logP = (np.sum(np.vstack((betaln(m_kl + r_nl + a, m_bar_kl + m - r_nl + b) - betaln(m_kl + a, m_bar_kl + b), \
                                      betaln(r_nl + a, m - r_nl + b) - betaln(a, b))), axis=1) + np.log(
                np.append(m, A_alpha)))  # k are rows and l are columns. Sum = log product over l.

            P = np.exp(logP - np.max(logP))  # Avoid underflow and turn into probabilities
            cum_post = np.cumsum(P / np.sum(P))
            new_assign = int(np.sum(
                np.random.rand() > cum_post))  # Calculate which cluster by finding where in the probability distribution rand() lands.
            z[n, :] = 0
            if new_assign == z.shape[1]:
                z = np.hstack((z, np.zeros((N, 1))))
            z[n, new_assign] = 1

            # z = z[np.nonzero(np.sum(z, axis = 0))] # But this new clustering can't have fewer clusters, since we compensated at the beginning.
            assert np.all(np.sum(z, axis=0) > 0)
        Z.append(
            z if not return_clustering_matrix else z @ z.T)  # for each z, the full clustering adjacency matrix is now just z @ z.T

    assert len(Z) == T
    Z = (Z[burn_in_buffer:] if burn_in_buffer is not None else Z[T // 2:])
    if sample_interval is not None: Z = Z[::sample_interval]
    return Z


def Gibbs_sample_torch(A, T, burn_in_buffer=None, sample_interval=None, seed=42, a=1, b=1, A_alpha=1,
                       return_clustering_matrix=False):
    """Perform one Gibbs sweep with a specified node order to yield one sampled clustering.

    Parameters
    ----------
    A : (2D ndarray) Adjacency matrix.
    T : (int) Number of Gibbs Sweeps.
    burn_in_buffer : (int) Number of initial samples to discard. Default is T//2.
    sample_interval : (int) The sweep interval with which to save clusterings.
    seed : (int) Seed used for random node permutations.
    a, b and A_alpha: float
        Parameters for the beta distribution prior for the cluster connectivities and concentration.
        a = b = 1 yields a uniform distribution.
        A_alpha is the total cluster concentration parameter.
    return_clustering_matrix: (bool) a list of full clustering adjacency matrices instead of partial one-hot encoded clustering matrices.

    Return
    ----------
    Z: (list) list of clusterings z sampled through iterations of Gibbs sweeps.
    """

    N = A.shape[0]
    A[torch.arange(N), torch.arange(N)] //= 2  # Assume nodes aren't connected to themselves.
    # z_f = np.zeros((N,N)) # z_full. Take all the memory I need, which is not that much.
    # z_f[:, 0] = 1 # Initialize everthing to be in the first cluster.
    z = torch.ones((N, 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    Z = []
    for t in tqdm(range(T), desc='Gibbs Sampling'):
        node_order = torch.randperm(
            N)  # I could also make the full (N,T) permutations matrix to exchange speed for memory
        for i, n in enumerate(node_order):
            # nn = node_order[node_order != n] # Permute hard in here as well because we can.
            nn_ = torch.arange(N)  # Another option that is a bit simpler and doesn't permute quite as hard.
            nn = nn_[nn_ != n]
            m = torch.sum(z[nn, :], axis=0)  # Could also use my old cluster representation, which is probably faster.
            K_ind = m.nonzero(as_tuple=True)[0]  # K_ind = m.nonzero(as_tuple=True)[0]
            K = len(K_ind)  # Number of clusters
            z, m = z[:, K_ind], m[K_ind]  # Fix potential empty clusters

            m_kl = z[nn, :].T @ A[nn][:, nn] @ z[nn, :]
            m_kl[torch.arange(K), torch.arange(
                K)] //= 2  # The edges are undirected and the previous line counts edges within a cluster twice as if directed. Compensate for that here.

            m_bar_kl = torch.outer(m, m) - torch.diag(m * (m + 1) / 2) - m_kl
            r_nl = (z[nn, :].T @ A[nn, n])  # node n connections to each cluster (l).

            # Calculate the big log posterior for the cluster allocation for node n. (k+1 possible cluster allocations)
            logP = torch.sum(
                torch.vstack((betaln(m_kl + r_nl + a, m_bar_kl + m - r_nl + b) - betaln(m_kl + a, m_bar_kl + b), \
                              betaln(r_nl + a, m - r_nl + b) - betaln(a, b))), axis=1) + torch.log(
                torch.hstack((m, torch.tensor([A_alpha]))))  # k are rows and l are columns. Sum = log product over l.

            P = torch.exp(logP - torch.max(logP))  # Avoid underflow and turn into probabilities
            cum_post = torch.cumsum(P / torch.sum(P), dim=0)
            new_assign = int(torch.sum(torch.rand(
                1) > cum_post))  # Calculate which cluster by finding where in the probability distribution rand() lands.
            z[n, :] = 0
            if new_assign == z.shape[1]:
                z = torch.hstack((z, torch.zeros((N, 1))))
            z[n, new_assign] = 1

            # z = z[np.nonzero(np.sum(z, axis = 0))] # But this new clustering can't have fewer clusters, since we compensated at the beginning.
            assert torch.all(torch.sum(z, axis=0) > 0)
        Z.append(
            z if not return_clustering_matrix else z @ z.T)  # for each z, the full clustering adjacency matrix is now just z @ z.T

    assert len(Z) == T
    Z = (Z[burn_in_buffer:] if burn_in_buffer is not None else Z[T // 2:])
    if sample_interval is not None: Z = Z[::sample_interval]
    return Z


def get_clustering_list(clustering_matrix):
    current_clustering_copy = torch.clone(clustering_matrix)
    clustering_list = torch.zeros(current_clustering_copy.size()[0])
    number_of_clusters = 1  # starting at the empty cluster
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


def clusters_all_index(clusters_all_tensor, specific_cluster_list):
    cluster_ind = torch.argwhere(torch.all(torch.eq(clusters_all_tensor, specific_cluster_list), dim=1) == 1)[0][0]
    return cluster_ind


def empiricalSampleDistribution(X1, n_nodes, net, numpy=False, log=True):
    clusters_all = allPermutations(n_nodes)
    clusters_all_tensor = torch.tensor(clusters_all + 1)
    # X1 = net.sample_forward(adjacency_matrix=adjacency_matrix, n_samples=n_samples)

    sample_posterior_counts = torch.zeros(len(clusters_all))

    for x in X1:
        x_c_list = get_clustering_list(net.get_matrices_from_state(x)[1])[0]

        cluster_ind = clusters_all_index(clusters_all_tensor, specific_cluster_list=x_c_list)
        sample_posterior_counts[cluster_ind] += 1

    sample_posterior_probs = sample_posterior_counts / torch.sum(sample_posterior_counts)
    if log:
        sample_posterior_probs = torch.log(sample_posterior_probs)
        assert -0.1 < torch.logsumexp(sample_posterior_probs, (0)) < 0.1

    return sample_posterior_probs.detach().numpy() if numpy else sample_posterior_probs


# %% MAIN
if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    check_gpu()
    N = 3
    a, b, A_alpha = 0.5, 0.5, 3
    adjacency_matrix, clusters = IRM_graph(A_alpha=A_alpha, a=a, b=b, N=N)
    cluster_idxs = clusterIndex(clusters)

    net2 = GraphNet(n_nodes=adjacency_matrix.size()[0], a=a, b=b, A_alpha=A_alpha, using_backward_model=True)
    net2.save()
    net2.load_forward()
    # net2.load_backward()
    X1 = net2.sample_forward(adjacency_matrix)
    losses2 = net2.train(X1, epochs=100)
    net2.plot_full_distribution(adjacency_matrix)

    net = GraphNet(n_nodes=adjacency_matrix.size()[0], a=a, b=b, A_alpha=A_alpha)
    X = net.sample_forward(adjacency_matrix)
    losses1 = net.train(X, epochs=100)
    net.plot_full_distribution(adjacency_matrix)

    plt.plot(losses1.detach().numpy(), label='Trajectory Balance Loss')
    plt.plot(losses2.detach().numpy(), label='L2 Error')
    plt.legend()
    plt.title('MSE Error per iteration')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Error')
    plt.show()
    # Nodes, loss functions, training time,
    # Compare with true IRM
