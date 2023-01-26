import os.path
import random
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import torch
import numpy as np

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class EagleDataset(Dataset):
    def __init__(self, data_path, mode="test", window_length=990, apply_onehot=True, with_cells=False,
                 with_cluster=False, n_cluster=20, normalize=False):
        """ Eagle dataset
        :param data_path: path to the dataset
        :param window_length: length of the temporal window to sample the simulation
        :param apply_onehot: Encode node type as onehot vector, see global variables to see what is what
        :param with_cells: Default is to return edges as pairs of indices, if True, return also the triangles (cells),
        useful for visualization
        :param with_cluster: Load the clustered indices of the nodes
        :param n_cluster: Number of cluster to use, 0 means no clustering / one node per cluster
        :param normalize: center mean and std of velocity/pressure field
        """
        super(EagleDataset, self).__init__()
        assert mode in ["train", "test", "valid"]

        self.window_length = window_length
        assert window_length <= 990, "window length must be smaller than 990"

        self.fn = data_path
        assert os.path.exists(self.fn), f"Path {self.fn} does not exist"

        self.apply_onehot = apply_onehot
        self.dataloc = []

        with open(f"Splits/{mode}.txt", "r") as f:
            for line in f.readlines():
                self.dataloc.append(os.path.join(self.fn, line.strip()))

        self.with_cells = with_cells
        self.with_cluster = with_cluster
        self.n_cluster = n_cluster
        self.mode = mode
        self.do_normalization = normalize
        self.length = 990

        if self.with_cluster:
            assert self.n_cluster in [0, 10, 20, 40, 30], f'Please, check if clustering has been computed offline ' \
                                                          f'for {self.n_cluster} nodes per cluster'

    def __len__(self):
        return len(self.dataloc)

    def __getitem__(self, item):
        mesh_pos, faces, node_type, t, velocity, pressure = get_data(self.dataloc[item], self.window_length, self.mode)
        faces = torch.from_numpy(faces).long()
        mesh_pos = torch.from_numpy(mesh_pos).float()
        velocity = torch.from_numpy(velocity).float()
        pressure = torch.from_numpy(pressure).float()
        edges = faces_to_edges(faces)  # Convert triangles to edges (pairs of indices)
        node_type = torch.from_numpy(node_type).long()

        if self.apply_onehot:
            node_type = one_hot(node_type, num_classes=9).squeeze(-2)

        if self.do_normalization:
            velocity, pressure = self.normalize(velocity, pressure)

        output = {'mesh_pos': mesh_pos,
                  'edges': edges,
                  'velocity': velocity,
                  'pressure': pressure,
                  'node_type': node_type}

        if self.with_cells:
            output['cells'] = faces

        if self.with_cluster:
            # If you want to use our clusters, you need to download them, and update this path
            cluster_path = self.dataloc[item].replace("Eagle_dataset", "Eagle_cluster")
            assert os.path.exists(cluster_path), f'Pre-computed cluster are not found, check path: {cluster_path}.\n' \
                                                 f'or update the path in the Dataloader.'
            if self.n_cluster == 0:
                clusters = torch.arange(mesh_pos.shape[1] + 1).view(1, -1, 1).repeat(velocity.shape[0], 1, 1)
            else:
                clusters = np.load(os.path.join(cluster_path, f"constrained_kmeans_{self.n_cluster}.npy"),
                                   mmap_mode='r')[t:t + self.window_length].copy()
                clusters = torch.from_numpy(clusters).long()
            output['cluster'] = clusters
        return output

    def normalize(self, velocity=None, pressure=None):
        if pressure is not None:
            pressure_shape = pressure.shape
            mean = torch.tensor([-0.8322, 4.6050]).to(pressure.device)
            std = torch.tensor([7.4013, 9.7232]).to(pressure.device)
            pressure = pressure.reshape(-1, 2)
            pressure = (pressure - mean) / std
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = torch.tensor([-0.0015, 0.2211]).to(velocity.device).view(-1, 2)
            std = torch.tensor([1.7970, 2.0258]).to(velocity.device).view(-1, 2)
            velocity = velocity.reshape(-1, 2)
            velocity = (velocity - mean) / std
            velocity = velocity.reshape(velocity_shape)

        return velocity, pressure

    def denormalize(self, velocity=None, pressure=None):
        if pressure is not None:
            pressure_shape = pressure.shape
            mean = torch.tensor([-0.8322, 4.6050]).to(pressure.device)
            std = torch.tensor([7.4013, 9.7232]).to(pressure.device)
            pressure = pressure.reshape(-1, 2)
            pressure = (pressure * std) + mean
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = torch.tensor([-0.0015, 0.2211]).to(velocity.device).view(-1, 2)
            std = torch.tensor([1.7970, 2.0258]).to(velocity.device).view(-1, 2)
            velocity = velocity.reshape(-1, 2)
            velocity = velocity * std + mean
            velocity = velocity.reshape(velocity_shape)

        return velocity, pressure


def get_data(path, window_length, mode):
    # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
    t = 0 if window_length == 990 else random.randint(0, 990 - window_length)
    t = 100 if mode != "train" and window_length != 990 else t
    data = np.load(os.path.join(path, 'sim.npz'), mmap_mode='r')

    mesh_pos = data["pointcloud"][t:t + window_length].copy()

    cells = np.load("/" + os.path.join(path, f"triangles.npy"))
    cells = cells[t:t + window_length]

    Vx = data['VX'][t:t + window_length].copy()
    Vy = data['VY'][t:t + window_length].copy()

    Ps = data['PS'][t:t + window_length].copy()
    Pg = data['PG'][t:t + window_length].copy()

    velocity = np.stack([Vx, Vy], axis=-1)
    pressure = np.stack([Ps, Pg], axis=-1)
    node_type = data['mask'][t:t + window_length].copy()

    return mesh_pos, cells, node_type, t, velocity, pressure


def faces_to_edges(faces):
    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, ::2]], dim=1)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=1)

    return unique_edges
