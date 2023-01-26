import torch
import torch.nn as nn
from Models.Base import MLP, multiHeadGAT
from torch_scatter import scatter_add

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class GraphAttention(nn.Module):
    def __init__(self, apply_noise=True, N=8, state_size=3, n_heads=4):
        """ Graph Attention Network
        :param apply_noise: whether to add noise to the input
        :param N: number of chained GAT layers
        :param state_size: number of channels in the input
        :param n_heads: number of heads in the GAT layers"""
        super(GraphAttention, self).__init__()

        self.apply_noise = apply_noise
        self.noise_std = 2e-2
        self.encoder = Encoder(state_size)
        self.processor = Processor(N, n_heads=n_heads)
        self.decoder = MLP(input_size=128, output_size=state_size, layer_norm=False)
        self.normalizer_output = Normalizer(state_size)

    def forward(self, mesh_pos, edges, state, node_type):
        """ Forward pass of the Graph Attention Network
        :param mesh_pos: tensor of shape (batch_size, sequence_length, num_nodes, 3)
        :param edges: tensor of shape (batch_size, sequence_length, num_edges, 2)
        :param state: tensor of shape (batch_size, sequence_length, num_nodes, state_size)
        :param node_type: tensor of shape (batch_size, sequence_length, num_nodes, 7)
        :return: state_hat, output_hat, target
        """
        if self.apply_noise:
            mask = torch.logical_or(node_type[:, 0, :, NODE_NORMAL] == 1, node_type[:, 0, :, NODE_OUTPUT] == 1)
            noise = torch.randn_like(state[:, 0]).to(state[:, 0].device) * self.noise_std
            state[:, 0][mask[:, 0]] = state[:, 0][mask[:, 0]] + noise[mask[:, 0]]

        state_hat, output_hat = [state[:, 0]], []
        target = state[:, 1:] - state[:, :-1]
        target = self.normalizer_output(target)

        for t in range(1, state.shape[1]):
            V, E = self.encoder(mesh_pos[:, t - 1], edges[:, t - 1], node_type[:, t - 1], state_hat[-1])
            V = self.processor(V, E, edges[:, t - 1])

            next_output = self.decoder(V)

            output_denormalized = self.normalizer_output.inverse(next_output)
            next_state = state_hat[-1] + output_denormalized

            mask = (node_type[:, t, :, NODE_INPUT] == 1) | (node_type[:, t, :, NODE_WALL] == 1) | (
                    node_type[:, t, :, NODE_DISABLE] == 1)

            next_state[mask, :] = state[:, t][mask, :]
            state_hat.append(next_state)
            output_hat.append(next_output)

        state_hat = torch.stack(state_hat, dim=1)
        output_hat = torch.stack(output_hat, dim=1)

        return state_hat, output_hat, target


class Encoder(nn.Module):
    def __init__(self, state_size):
        super(Encoder, self).__init__()

        self.normalize_edges = Normalizer(3)
        self.normalize_nodes = Normalizer(9 + state_size)

        self.fv = MLP(input_size=9 + state_size)
        self.fe = MLP(input_size=3)

    def forward(self, mesh_pos, edges, node_type, velocity):
        # Get nodes embeddings
        V = torch.cat([velocity, node_type], dim=-1)
        V = self.fv(self.normalize_nodes(V))
        # Get edges attr
        senders = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
        receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))

        distance = senders - receivers
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)

        E = self.fe(self.normalize_edges(E))

        return V, E


class Processor(nn.Module):
    def __init__(self, N=15, n_heads=4):
        super(Processor, self).__init__()
        self.gat = nn.ModuleList([multiHeadGAT(node_size=128, output_size=128, n_heads=n_heads) for _ in range(N)])

    def forward(self, V, E, edges):
        for gat in self.gat:
            v = gat(V, E, edges)
            V = V + v
        return V


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.delta = MLP(input_size=128, output_size=3, layer_norm=False)

    def forward(self, V, velocity):
        output = self.delta(V)
        velocity = velocity + output[..., :2]
        pressure = output[..., -1:]
        return velocity, pressure, output


class Normalizer(nn.Module):
    def __init__(self, input_size):
        super(Normalizer, self).__init__()
        self.accumulation = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.accumulation_squarred = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.std = nn.Parameter(torch.ones(input_size), requires_grad=False)
        self.max_accumulation = 1e7
        self.count = 0
        self.input_size = input_size

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])
        if self.training is True:
            if self.count < self.max_accumulation:
                self.count += x.shape[0]
                self.accumulation.data = self.accumulation.data + torch.mean(x, dim=(0, 1))
                self.accumulation_squarred.data = self.accumulation_squarred.data + torch.mean(x ** 2, dim=(0, 1))
                self.mean.data = self.accumulation / (self.count + 1e-8)
                self.std.data = torch.sqrt(self.accumulation_squarred / (self.count + 1e-8) - self.mean.data ** 2)
        return (x.reshape(*original_shape) - self.mean) / (self.std + 1e-8)

    def inverse(self, x):
        return x * self.std + self.mean
