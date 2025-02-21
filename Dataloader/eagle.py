"""Dataloader for the Eagle dataset

The dataloader outputs a dictionary with the following keys:
- mesh_pos: (T, N, 2) tensor of the mesh position at each time step
- edges: (E, 2) tensor of the edges in the mesh, defined by the indices of the nodes
- velocity: (T, N, 2) tensor of the velocity field at each time step
- pressure: (T, N, 2) tensor of the pressure field at each time step
- node_type: (T, N) tensor of the type of each node. The types are integer labels and
  follows the same encoding as MeshGraphNet (NORMAL=0, INPUT=4, OUTPUT=5, WALL=6,
  DISABLE=2)
"""

import torch
import random
import numpy as np
from pathlib import Path
from torch import Tensor
from numpy.typing import NDArray
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from typing import Union, List, Tuple, Optional, Dict


class EagleDataset(Dataset):
    """Eagle dataset.

    Parameters:
        data_path (Union[Path, str]): Path to the dataset (root folder containing ``
            Spl``, ``Cre`` and ``Tri`` folders).
        split (str): Split to load (``train``, ``test`` or ``valid``). The dataloader
            will load the corresponding file from the ``splits`` folder.
        window_length (int): Length of the temporal window to sample the simulation.
            The default value is 990 (full simulation). Simulations will be cropped
            to this length (see :py:function:`_load_from_npz`).
        type_as_onehot (bool): If True, node type labels are encoded as one-hot vectors.
            Default is :py:`True`.
        with_cells (bool): If :py:`True`, the dataloader will return the cells
            (triangles) of the mesh. Default is :py:`False`. This is mainly used for
            visualization.
        n_cluster (int): Number of node per cluster to use. If -1, no cluster are used
            (the ``cluster`` key will not be present in the output). If 1, then
            each node is its own cluster. Otherwise, the dataloader will load the
            clusters from the ``cluster_path`` folder. Default is -1.
        normalized (bool): If :py:`True`, the velocity and pressure fields are
            normalized to have zero mean and unit variance. Default is :py:`False`.
        cluster_path (Optional[Union[Path, str]]): Path to the folder containing the
            clusters. Make sure this path is correct if you want to use the clusters.

    Attributes:
        w_len (int): Length of the temporal window to sample the simulation.
        split (str): Split to load (``train``, ``test`` or ``valid``).
        n_cluster (int): Number of node per cluster to use.
        data_path (Path): Path to the dataset.
        _with_cells (bool): If :py:`True`, the dataloader will return the cells
            (triangles) of the mesh.
        cluster_path (Path): Path to the folder containing the clusters.
        ep_paths (List[Path]): List of paths to the simulation files.
        _type_as_onehot (bool): If :py:`True`, node type labels are encoded as one-hot
            vectors.
        _use_normalized (bool): If :py:`True`, the velocity and pressure fields are
            normalized to have zero mean and unit variance.
        pressure_mean (Tensor): Mean of the pressure field.
        pressure_std (Tensor): Standard deviation of the pressure field.
        velocity_mean (Tensor): Mean of the velocity field.
        velocity_std (Tensor): Standard deviation of the velocity field.

    Note:
        The dataset statistics are hard-coded in the class. They might not correspond
        exactly to the statistics of the last version of Eagle dataset (as some
        simulations were discarded during the project). Use these values when
        starting from our pre-trained model, but consider computing the statistics
        on the full dataset if you want to train from scratch.
    """

    w_len: int
    split: str
    n_cluster: int
    data_path: Path
    _with_cells: bool
    cluster_path: Path
    ep_paths: List[Path]
    _type_as_onehot: bool
    _use_normalized: bool
    pressure_mean: Tensor
    pressure_std: Tensor
    velocity_mean: Tensor
    velocity_std: Tensor

    def __init__(
        self,
        data_path: Union[Path, str],
        split: str,
        window_length: int = 990,
        type_as_onehot: bool = True,
        with_cells: bool = False,
        n_cluster: int = -1,
        normalized: bool = False,
        cluster_path: Optional[Union[Path, str]] = None,
    ) -> None:
        super(EagleDataset, self).__init__()

        self.w_len = window_length
        assert self.w_len <= 990, "window length must be smaller than 990"

        self.data_path = data_path if isinstance(data_path, Path) else Path(data_path)
        assert self.data_path.exists()
        if cluster_path is not None:
            self.cluster_path = (
                cluster_path if isinstance(cluster_path, Path) else Path(cluster_path)
            )
            assert self.data_path.exists()

        with open(f"splits/{split}.txt", "r") as f:
            self.ep_paths = [self.data_path / l.strip() for l in f.readlines()]

        self.n_cluster = n_cluster
        self._type_as_onehot = type_as_onehot
        self._with_cells = with_cells
        self.split = split
        self._use_normalized = normalized

        assert self.n_cluster in [-1, 1, 10, 20, 40, 30], "Unknown number of clusters"

        self.pressure_mean = torch.tensor([-0.8322, 4.6050]).view(-1, 2)
        self.pressure_std = torch.tensor([7.4013, 9.7232]).view(-1, 2)
        self.velocity_mean = torch.tensor([-0.0015, 0.2211]).view(-1, 2)
        self.velocity_std = torch.tensor([1.7970, 2.0258]).view(-1, 2)

    def __len__(self):
        return len(self.ep_paths)

    def __getitem__(self, item):
        mesh_pos, faces, node_type, t, velocity, pressure = _load_from_npz(
            self.ep_paths[item], self.w_len, self.split
        )
        faces = torch.from_numpy(faces).long()
        mesh_pos = torch.from_numpy(mesh_pos).float()
        velocity = torch.from_numpy(velocity).float()
        pressure = torch.from_numpy(pressure).float()
        edges = _faces_to_edges(faces)  # Convert triangles to edges (pairs of indices)
        node_type = torch.from_numpy(node_type).long()

        if self._type_as_onehot:
            node_type = one_hot(node_type, num_classes=9).squeeze(-2)

        if self._use_normalized:
            velocity, pressure = self.normalize(velocity, pressure)

        output = {
            "mesh_pos": mesh_pos,
            "edges": edges,
            "velocity": velocity,
            "pressure": pressure,
            "node_type": node_type,
        }

        if self._with_cells:
            output["cells"] = faces

        if self.n_cluster != -1:
            cluster_path = self.cluster_path / self.ep_paths[item].relative_to(
                self.data_path
            )

            if self.n_cluster == 1:
                clusters = (
                    torch.arange(mesh_pos.shape[1] + 1)
                    .view(1, -1, 1)
                    .repeat(velocity.shape[0], 1, 1)
                )
            else:
                clusters = np.load(
                    cluster_path / f"constrained_kmeans_{self.n_cluster}.npy",
                    mmap_mode="r",
                )[t : t + self.w_len].copy()
                clusters = torch.from_numpy(clusters).long()
            output["cluster"] = clusters
        return output

    def normalize(self, velocity: Tensor, pressure: Tensor) -> Tuple[Tensor, Tensor]:
        """Normalize the velocity and pressure fields use dataset statistics.

        Parameters:
            velocity (Tensor): Velocity field to normalize. Shape is (*, 2).
            pressure (Tensor): Pressure field to normalize. Shape is (*, 2).

        Returns:
            Tuple[Tensor, Tensor]: Normalized velocity and pressure fields.
        """
        p_mean, v_mean, p_std, v_std = self._stat_to_device(pressure.device, velocity.device)

        p_shape, v_shape = pressure.shape, velocity.shape
        pressure, velocity = pressure.reshape(-1, 2), velocity.reshape(-1, 2)
        pressure = (pressure - p_mean) / p_std
        velocity = (velocity - v_mean) / v_std
        return velocity.reshape(v_shape), pressure.reshape(p_shape)

    def denormalize(self, velocity: Tensor, pressure: Tensor) -> Tuple[Tensor, Tensor]:
        """Denormalize the velocity and pressure fields use dataset statistics.

        Parameters:
            velocity (Tensor): Velocity field to denormalize. Shape is (*, 2).
            pressure (Tensor): Pressure field to denormalize. Shape is (*, 2).

        Returns:
            Tuple[Tensor, Tensor]: Denormalized velocity and pressure fields.
        """
        p_mean, v_mean, p_std, v_std = self._stat_to_device(pressure.device, velocity.device)

        p_shape, v_shape = pressure.shape, velocity.shape
        pressure, velocity = pressure.reshape(-1, 2), velocity.reshape(-1, 2)
        pressure = (pressure * p_std) + p_mean
        velocity = (velocity * v_std) + v_mean
        return velocity.reshape(v_shape), pressure.reshape(p_shape)

    def _stat_to_device(
        self, p_device: torch.device, v_device: torch.device
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Send the dataset statistics to the given devices.

        Parameters:
            p_device (torch.device): Device to send the pressure statistics.
            v_device (torch.device): Device to send the velocity statistics.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Pressure and velocity statistics
        """
        return (
            self.pressure_mean.to(p_device),
            self.velocity_mean.to(v_device),
            self.pressure_std.to(p_device),
            self.velocity_std.to(v_device)
        )

def _load_from_npz(
    path: Path, window_length: int, split: str
) -> Tuple[NDArray, NDArray, NDArray, int, NDArray, NDArray]:
    """Load the simulation data from the npz file.

    Note:
        The function also crop the simulated sequence to the given window length.
        If the selected split is ``train``, the starting point is sampled randomly in
        the sequence, otherwise it is fixed to 100 (to ensure repeatability).

    Parameters:
        path (Path): Path to the npz file.
        window_length (int): Length of the temporal window to sample the simulation.
        split (str): Split to load (``train``, ``test`` or ``valid``).

    Returns:
        mesh_pos (NDArray): Mesh node positions. Shape is (T, N, 2).
        cells (NDArray): Mesh cells (triangles). Shape is (T, N, 3).
        node_type (NDArray): Node type labels. Shape is (T, N).
        t (int): Starting point of the sequence.
        velocity (NDArray): Velocity field. Shape is (T, N, 2).
        pressure (NDArray): Pressure field. Shape is (T, N, 2).
    """
    t = 0 if window_length == 990 else random.randint(0, 990 - window_length)
    t = 100 if split != "train" and window_length != 990 else t
    data = np.load(path / "sim.npz", mmap_mode="r")

    mesh_pos = data["pointcloud"][t : t + window_length].copy()

    cells = np.load(path / "triangles.npy")
    cells = cells[t : t + window_length]

    Vx = data["VX"][t : t + window_length].copy()
    Vy = data["VY"][t : t + window_length].copy()

    Ps = data["PS"][t : t + window_length].copy()
    Pg = data["PG"][t : t + window_length].copy()

    velocity = np.stack([Vx, Vy], axis=-1)
    pressure = np.stack([Ps, Pg], axis=-1)
    node_type = data["mask"][t : t + window_length].copy()

    return mesh_pos, cells, node_type, t, velocity, pressure


def _faces_to_edges(faces: Tensor) -> Tensor:
    """Convert the faces (triangles) to edges (pairs of indices).

    Edges are directional, i.e. (i, j) and (j, i) are considered different edges, so
    each face will generate 6 edges.

    Parameters:
        faces (Tensor): Faces of the mesh. Shape is (T, N, 3). Each face is defined by
            the indices of the nodes.

    Returns:
        Tensor: Edges of the mesh. Shape is (E, 2). Each edge is defined by the indices
            of the nodes.
    """
    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, ::2]], dim=1)

    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)

    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=1)

    return unique_edges


def collate(x_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Aggregate a list of samples into a batch and pad the sequences to account for
    different number of nodes and edges in each sample.

    The function adds ghost nodes and edges to each sample so that they all have the
    same shape. The ghost nodes are added at the end of the sequence, and the ghost
    edges are added at the end of the edge list.

    Parameters:
        x_list (List[Dict[str, torch.Tensor]]): List of samples to aggregate.

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing the aggregated samples.
    """
    # Find the maximum number of nodes and edges in the batch
    N_max = max([x["mesh_pos"].shape[-2] for x in x_list])
    E_max = max([x["edges"].shape[-2] for x in x_list])
    C_max = max([x["cluster"].shape[-2] for x in x_list])

    for batch, x in enumerate(x_list):
        # This step add fantom nodes to reach N_max + 1 nodes
        for key in ["mesh_pos", "velocity", "pressure"]:
            tensor = x[key]
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)

        tensor = x["node_type"]
        T, N, S = tensor.shape
        x["node_type"] = torch.cat([tensor, 2 * torch.ones(T, N_max - N + 1, S)], dim=1)

        x["cluster_mask"] = torch.ones_like(x["cluster"])
        x["cluster_mask"][x["cluster"] == -1] = 0
        x["cluster"][x["cluster"] == -1] = N_max

        if x["cluster"].shape[1] < C_max:
            c = x["cluster"].shape[1]
            x["cluster"] = torch.cat(
                [
                    x["cluster"],
                    N_max
                    * torch.ones(
                        x["cluster"].shape[0], C_max - c, x["cluster"].shape[-1]
                    ),
                ],
                dim=1,
            )
            x["cluster_mask"] = torch.cat(
                [
                    x["cluster_mask"],
                    torch.zeros(
                        x["cluster_mask"].shape[0], C_max - c, x["cluster"].shape[-1]
                    ),
                ],
                dim=1,
            )

        edges = x["edges"]
        T, E, S = edges.shape
        x["edges"] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)
        x["mask"] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)

    output = {key: torch.empty(1) for key in x_list[0].keys()}
    for key in output.keys():
        output[key] = torch.stack([x[key] for x in x_list], dim=0)
    return output


if __name__ == "__main__":

    d = EagleDataset(
        data_path=Path("/beegfs/scratch/user/sjanny/eagle_dataset/"),
        cluster_path=Path("/beegfs/scratch/user/sjanny/eagle_dataset/clusters"),
        split="valid",
        window_length=990,
    )

    p, v = [], []
    from tqdm import tqdm

    for i in tqdm(range(len(d))):
        x = d[i]
        p.append(x["pressure"].reshape(-1, 2))
        v.append(x["velocity"].reshape(-1, 2))
    p = torch.cat(p, dim=0)
    v = torch.cat(v, dim=0)
    print("Velocity mean:", v.mean(dim=0))
    print("Velocity std:", v.std(dim=0))
    print("Pressure mean:", p.mean(dim=0))
    print("Pressure std:", p.std(dim=0))
