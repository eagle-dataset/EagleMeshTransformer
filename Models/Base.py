import torch.nn as nn
import torch
from torch_scatter import scatter_sum


class MLP(nn.Module):
    def __init__(self, input_size, output_size=128, layer_norm=True, n_hidden=2, hidden_size=128):
        super(MLP, self).__init__()
        if hidden_size == 0:
            f = [nn.Linear(input_size, output_size)]
        else:
            f = [nn.Linear(input_size, hidden_size), nn.ReLU()]
            h = 1
            for i in range(h, n_hidden):
                f.append(nn.Linear(hidden_size, hidden_size))
                f.append(nn.ReLU())
            f.append(nn.Linear(hidden_size, output_size))
            if layer_norm:
                f.append(nn.LayerNorm(output_size))

        self.f = nn.Sequential(*f)

    def forward(self, x):
        return self.f(x)


class GNN(nn.Module):
    def __init__(self, n_hidden=2, node_size=128, edge_size=128, output_size=None, layer_norm=False):
        super(GNN, self).__init__()
        output_size = output_size or node_size
        self.f_edge = MLP(input_size=edge_size + node_size * 2, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=edge_size)
        self.f_node = MLP(input_size=edge_size + node_size, n_hidden=n_hidden, layer_norm=layer_norm,
                          output_size=output_size)

    def forward(self, V, E, edges):
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        edge_embeddings = self.f_edge(edge_inpt)

        col = edges[..., 0].unsqueeze(-1).repeat(1, 1, edge_embeddings.shape[-1])
        edge_sum = scatter_sum(edge_embeddings, col, dim=-2)

        node_inpt = torch.cat([V, edge_sum], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


class multiHeadGAT(nn.Module):
    def __init__(self, node_size, output_size, n_heads):
        super(multiHeadGAT, self).__init__()
        assert output_size % n_heads == 0, "output_size must be divisible by n_heads"
        self.n_heads = n_heads
        self.gat = nn.ModuleList([GAT(node_size, output_size // n_heads) for _ in range(n_heads)])

    def forward(self, V, E, edges):
        heads = [gat(V, E, edges) for gat in self.gat]
        return torch.cat(heads, dim=-1)


class GAT(nn.Module):
    def __init__(self, node_size, output_size):
        super(GAT, self).__init__()
        self.f_edge = nn.Linear(in_features=node_size, out_features=output_size, bias=False)
        self.attention = nn.Sequential(nn.Linear(in_features=output_size * 2 + 128, out_features=1),
                                       nn.LeakyReLU(0.2))

    def forward(self, V, E, edges):
        senders = torch.gather(V, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, V.shape[-1]))
        receivers = torch.gather(V, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, V.shape[-1]))

        h_sender = self.f_edge(senders)
        h_receiver = self.f_edge(receivers)

        attention = self.attention(torch.cat([h_sender, h_receiver, E], dim=-1))
        attention = torch.exp(attention - torch.max(attention, dim=1, keepdim=True)[0])
        col = edges[..., 0].unsqueeze(-1)

        numerator = scatter_sum(attention * h_sender, col.repeat(1, 1, h_sender.shape[-1]), dim=-2)
        denominator = scatter_sum(attention, col.repeat(1, 1, attention.shape[-1]), dim=-2)
        h = numerator / (denominator + 1e-8)

        return h


class Normalizer(nn.Module):
    def __init__(self, input_size):
        # This module accumulates stats on the data. It is inspired from the tensorflow code
        # of MeshGraphNet, and this model seems to work better with this instead of a layernorm
        # or manual normalization... Weird...
        super(Normalizer, self).__init__()
        self.accumulation = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.accumulation_squarred = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(input_size), requires_grad=False)
        self.std = nn.Parameter(torch.zeros(input_size), requires_grad=False)
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
