import torch.nn as nn
import torch
from Models.Base import MLP, GNN

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class GraphViT(nn.Module):
    def __init__(self, state_size, w_size=512, n_attention=4, nb_gn=4, n_heads=4):
        super(GraphViT, self).__init__()
        pos_start = -3
        pos_length = 8
        self.encoder = Encoder(nb_gn, state_size, pos_length)
        self.graph_pooling = GraphPooling(w_size, pos_length=pos_length)
        self.graph_retrieve = GraphRetrieveSimple(w_size, pos_length, state_size)
        self.attention = nn.ModuleList([AttentionBlock_PreLN(w_size, pos_length, n_heads) for _ in range(n_attention)])
        self.ln = nn.LayerNorm(w_size)

        self.noise_std = 0.0
        self.positional_encoder = Positional_Encoder(pos_start, pos_length)

    def forward(self, mesh_pos, edges, state, node_type, clusters, clusters_mask, apply_noise=False):
        if apply_noise:
            # Following MGN, this add noise to the input. Better results are obtained with longer windows and no noise
            mask = torch.logical_or(node_type[:, 0, :, NODE_NORMAL] == 1, node_type[:, 0, :, NODE_OUTPUT] == 1)
            noise = torch.randn_like(state[:, 0]).to(state[:, 0].device) * self.noise_std
            state[:, 0][mask] = state[:, 0][mask] + noise[mask]

        state_hat, output_hat = [state[:, 0]], []
        target = []

        for t in range(1, state.shape[1]):
            mesh_posenc, cluster_posenc = self.positional_encoder(mesh_pos[:, t - 1], clusters[:, t - 1],
                                                                  clusters_mask[:, t - 1])

            V, E = self.encoder(mesh_pos[:, t - 1], edges[:, t - 1], state_hat[-1], node_type[:, t - 1], mesh_posenc)
            W = self.graph_pooling(V, clusters[:, t - 1], mesh_posenc, clusters_mask[:, t - 1])

            # This attention_mask deals with the ghost nodes needed to batch multiple simulations
            attention_mask = clusters_mask[:, t - 1].sum(-1, keepdim=True) == 0
            attention_mask = attention_mask.unsqueeze(1).repeat(1, len(self.attention), 1, W.shape[1]).view(-1, W.shape[1], W.shape[1])
            attention_mask[:, torch.eye(W.shape[1], dtype=torch.bool)] = False
            attention_mask = attention_mask.transpose(-1, -2)

            for i, a in enumerate(self.attention):
                W = a(W, attention_mask, cluster_posenc)
            W = self.ln(W)

            next_output = self.graph_retrieve(W, V, clusters[:, t - 1], mesh_posenc, edges[:, t - 1], E)
            next_state = state_hat[-1] + next_output

            target.append(state[:, t] - state_hat[-1])

            # Following MGN, we force the boundary conditions at each steps
            mask = torch.logical_or(node_type[:, t, :, NODE_INPUT] == 1, node_type[:, t, :, NODE_WALL] == 1)
            mask = torch.logical_or(mask, node_type[:, t, :, NODE_DISABLE] == 1)
            next_state[mask, :] = state[:, t][mask, :]

            state_hat.append(next_state)
            output_hat.append(next_output)

        velocity_hat = torch.stack(state_hat, dim=1)
        output_hat = torch.stack(output_hat, dim=1)

        target = torch.stack(target, dim=1)
        return velocity_hat, output_hat, target


class AttentionBlock_PreLN(nn.Module):
    def __init__(self, w_size, pos_length, n_heads):
        super(AttentionBlock_PreLN, self).__init__()
        self.ln1 = nn.LayerNorm(w_size)

        embed_dim = w_size + 4 * pos_length

        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, w_size)
        self.ln2 = nn.LayerNorm(w_size)
        self.mlp = MLP(input_size=w_size, n_hidden=1, output_size=w_size, hidden_size=w_size, layer_norm=False)

    def forward(self, W, attention_mask, posenc):
        W1 = self.ln1(W)
        W1_posenc = torch.cat([W1, posenc], dim=-1)
        W2 = self.attention(W1_posenc, W1_posenc, W1_posenc, attn_mask=attention_mask)[0]
        W3 = W + self.linear(W2)

        W4 = self.ln2(W3)
        W5 = self.mlp(W4)

        W6 = W3 + W5
        return W6


class GraphPooling(nn.Module):
    def __init__(self, w_size, pos_length):
        super(GraphPooling, self).__init__()
        input_size = 128 + pos_length * 8

        self.rnn_pooling = nn.GRU(input_size=input_size, hidden_size=w_size, batch_first=True)
        self.linear_rnn = MLP(input_size=w_size, output_size=w_size, n_hidden=1, layer_norm=False)

    def forward(self, V, clusters, positional_encoding, cluster_mask):
        pos_by_cluster = torch.gather(positional_encoding, -2, clusters.reshape(clusters.shape[0], -1, 1).repeat(1, 1,
                                                                                                                 positional_encoding.shape[
                                                                                                                     -1]))
        pos_features = pos_by_cluster.reshape(*clusters.shape, -1)

        V_by_cluster = torch.gather(V, -2, clusters.reshape(clusters.shape[0], -1, 1).repeat(1, 1, V.shape[-1]))
        V_by_cluster = V_by_cluster.reshape(*clusters.shape, -1)

        inpt_by_cluster = torch.cat([V_by_cluster, pos_features], dim=-1)

        B, C, N, S = inpt_by_cluster.shape
        output, h = self.rnn_pooling(inpt_by_cluster.reshape(B * C, N, S))
        indices = (cluster_mask.sum(-1).long() - 1).reshape(B * C)
        indices[indices == -1] = output.shape[-2] - 1
        w = torch.gather(output, 1, indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, output.shape[-1]))

        w = self.linear_rnn(w)
        W = w.reshape(B, C, -1)

        return W


class GraphRetrieveSimple(nn.Module):
    def __init__(self, w_size, pos_length, state_size):
        pos_size = pos_length * 8
        super(GraphRetrieveSimple, self).__init__()
        node_size = w_size + 128 + pos_size
        self.gnn = GNN(node_size=node_size, output_size=128)
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, state_size)
        )

    def forward(self, W, V, clusters, positional_encoding, edges, E):
        B, N, S = V.shape
        C = W.shape[1]
        K = clusters.shape[-1]

        W = W.unsqueeze(-2).repeat(1, 1, K, 1).view(B, C * K, -1)
        W = W.scatter(-2, clusters.reshape(B, -1, 1).repeat(1, 1, W.shape[-1]), W)
        W = W[:, :N]

        nodes = torch.cat([V, W, positional_encoding], dim=-1)
        nodes, _ = self.gnn(nodes, E, edges)
        final_state = self.final_mlp(nodes)
        return final_state


class Encoder(nn.Module):
    def __init__(self, nb_gn=4, state_size=3, pos_length=7):
        super(Encoder, self).__init__()
        self.encoder_node = MLP(input_size=9 + state_size, output_size=128, n_hidden=1, layer_norm=False)
        self.encoder_edge = MLP(input_size=3, output_size=128, n_hidden=1, layer_norm=False)

        node_size = 128 + pos_length * 8
        self.encoder_gn = nn.ModuleList(
            [GNN(node_size=node_size, edge_size=128, output_size=128, layer_norm=True) for _ in
             range(nb_gn)])

    def forward(self, mesh_pos, edges, states, node_type, pos_enc):
        V = torch.cat([states, node_type], dim=-1)

        senders = torch.gather(mesh_pos, -2, edges[..., 0].unsqueeze(-1).repeat(1, 1, 2))
        receivers = torch.gather(mesh_pos, -2, edges[..., 1].unsqueeze(-1).repeat(1, 1, 2))

        distance = senders - receivers
        norm = torch.sqrt((distance ** 2).sum(-1, keepdims=True))
        E = torch.cat([distance, norm], dim=-1)

        V = self.encoder_node(V)
        E = self.encoder_edge(E)

        for i in range(len(self.encoder_gn)):
            inpt = torch.cat([V, pos_enc], dim=-1)
            v, e = self.encoder_gn[i](inpt, E, edges)
            V = V + v
            E = E + e

        return V, E


class Positional_Encoder(nn.Module):
    def __init__(self, pos_start, pos_length):
        super(Positional_Encoder, self).__init__()
        self.pos_length = pos_length
        self.pos_start = pos_start

    def forward(self, mesh_pos, clusters, cluster_mask):
        B, N, _ = mesh_pos.shape
        _, K, C = clusters.shape

        meshpos_by_cluster = torch.gather(mesh_pos, -2, clusters.reshape(B, -1, 1).repeat(1, 1, 2))
        meshpos_by_cluster = meshpos_by_cluster.reshape(*clusters.shape, -1)

        clusters_centers = meshpos_by_cluster.sum(dim=-2)
        clusters_centers = clusters_centers / (cluster_mask.sum(-1, keepdim=True) + 1e-8)

        distances_to_cluster = clusters_centers.unsqueeze(-2) - meshpos_by_cluster
        pos_embeddings = self.embed(distances_to_cluster)
        S = pos_embeddings.shape[-1]
        pos_embeddings = pos_embeddings.reshape(B, -1, S)
        relative_positions = pos_embeddings.scatter(-2, clusters.reshape(B, -1, 1).repeat(1, 1, S),
                                                    pos_embeddings.view(B, -1, S))
        relative_positions = relative_positions[:, :N]

        nodes_embedding = torch.cat([self.embed(mesh_pos), relative_positions], dim=-1)

        return nodes_embedding, self.embed(clusters_centers)

    def embed(self, pos):
        original_shape = pos.shape
        pos = pos.reshape(-1, original_shape[-1])
        index = torch.arange(self.pos_start, self.pos_start + self.pos_length, device=pos.device)
        index = index.float()
        freq = 2 ** index * torch.pi
        cos_feat = torch.cos(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        sin_feat = torch.sin(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        embedding = torch.cat([cos_feat, sin_feat], dim=-1)
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding
