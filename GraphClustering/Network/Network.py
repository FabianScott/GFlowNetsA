import numpy as np
import torch
import torch.nn as nn

class GraphNet:
    def __init__(self,
                 n_layers=2,
                 n_hidden=32,
                 gamma=0.5,
                 epochs=100,
                 lr=0.005,
                 # decay_steps=10000,
                 # decay_rate=0.8,
                 n_samples=1000
                 ):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gamma = gamma  # weighting of random sampling if applied
        self.epochs = epochs
        self.lr = lr
        self.n_samples = n_samples
        self.model = self.create_model()

    def create_model(self):
        # Define the layers of the neural network
        layers = []
        input_size = 0
        hidden_sizes, output_sizes = [], []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        output_layers = nn.ModuleList([nn.Softmax(prev_size, output_size) for output_size in output_sizes])
        layers.append(output_layers)

        # Define the forward function of the neural network
        def forward(x):
            for layer in layers:
                x = layer(x)
            outputs = [output_layer(x) for output_layer in output_layers]
            return outputs

        # Create an instance of the neural network and return it
        net = nn.Module()
        net.forward = forward
        return net


if '__name__' == '__main__':
    import torch
    x = torch.rand(5, 3)
    print(x)
