import os.path
import random
from torch.utils.data import Dataset
import numpy as np
import torch


class EagleDataset(Dataset):
    def __init__(self, data_path, mode="test", window_length=990, with_mesh=False):
        """ Eagle dataloader for images
        :param data_path: path to dataset (grid)
        :param mode: train, test or valid set
        :param window_length: length of the temporal window to sample the simulation
        :param with_mesh: load the irregular mesh, useful for evaluation purposes
        """

        super(EagleDataset, self).__init__()
        assert mode in ["train", "test", "valid"]

        self.window_length = window_length
        assert window_length <= 990, "window length must be smaller than 990"

        self.fn = data_path
        assert os.path.exists(self.fn), f"Path {self.fn} does not exist"

        self.dataloc = []
        with open(f"Splits/{mode}.txt", "r") as f:
            for line in f.readlines():
                self.dataloc.append(os.path.join(self.fn, line.strip()))

        self.mode = mode
        self.length = 990
        self.with_mesh = with_mesh

    def __len__(self):
        return len(self.dataloc)

    def __getitem__(self, item):
        # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
        t = random.randint(1, 990 - self.window_length) if self.window_length != 990 else 1
        t = 550 if self.mode in ["test", "valid"] and self.window_length != 990 else t

        states = np.load(os.path.join(self.dataloc[item], "states.npy"), mmap_mode='r')
        mask = np.load(os.path.join(self.dataloc[item], "pixel_type.npy"), mmap_mode='r')

        states = torch.from_numpy(states[t:t + self.window_length].copy()).float()

        output = {'states': self.normalize(states),
                  'mask': mask.copy(),
                  'example': torch.tensor((int(self.dataloc[item].split("/")[-2]),)), }

        if self.with_mesh:
            path = self.dataloc[item].replace("_img", "")
            assert os.path.exists(path), f"Can not find mesh files in {path}, please check the path in the dataloader"
            data = np.load(os.path.join(path, 'sim.npz'), mmap_mode='r')
            mesh_pos = data["pointcloud"][t:t + self.window_length].copy()
            Vx = data['VX'][t:t + self.window_length].copy()
            Vy = data['VY'][t:t + self.window_length].copy()
            Ps = data['PS'][t:t + self.window_length].copy()
            Pg = data['PG'][t:t + self.window_length].copy()
            velocity = np.stack([Vx, Vy], axis=-1)
            pressure = np.stack([Ps, Pg], axis=-1)
            node_type = data['mask'][t:t + self.window_length].copy()

            output['mesh_pos'] = mesh_pos
            output['mesh_velocity'] = velocity
            output['mesh_pressure'] = pressure
            output['mesh_node_type'] = node_type

        return output

    def normalize(self, state):
        shape = state.shape

        state = state.reshape(-1, 4)
        mean = torch.tensor([-0.0147, 0.2125, -0.5327, 3.7694]).to(state.device)
        std = torch.tensor([1.5943, 1.8824, 6.3553, 9.0565]).to(state.device)

        state = (state - mean) / std
        return state.reshape(shape)

    def denormalize(self, state):
        shape = state.shape

        state = state.reshape(-1, 4)
        mean = torch.tensor([-0.0147, 0.2125, -0.5327, 3.7694]).to(state.device)
        std = torch.tensor([1.5943, 1.8824, 6.3553, 9.0565]).to(state.device)

        state = state * std + mean
        return state.reshape(shape)


def grid2mesh(velocity_grid, pressure_grid, mesh_pos):
    """Project back the grid to the mesh. For evaluation."""
    Xmin, Xmax = -2.5, 2.5
    Ymin, Ymax = -1.7, 1.5

    LENGTH = 256
    HEIGHT = 128

    x, y = np.linspace(Xmin, Xmax, LENGTH), np.linspace(Ymax, Ymin, HEIGHT)
    step_x, step_y = x[1] - x[0], y[1] - y[0]
    velocity_mesh, pressure_mesh = [], []

    velocity_grid = np.flip(velocity_grid, axis=1)
    pressure_grid = np.flip(pressure_grid, axis=1)
    if type(mesh_pos) == torch.Tensor:
        mesh_pos = mesh_pos.cpu().numpy()

    for t in range(mesh_pos.shape[0]):
        index_x = ((mesh_pos[t, :, 0] - Xmin + step_x / 2) // step_x).astype(int)
        index_y = ((mesh_pos[t, :, 1] - Ymin + step_y / 2) // (-step_y)).astype(int)

        v_grid = velocity_grid[t]
        p_grid = pressure_grid[t]
        v = v_grid[index_y, index_x]
        p = p_grid[index_y, index_x]
        velocity_mesh.append(v)
        pressure_mesh.append(p)

    velocity_mesh = torch.from_numpy(np.stack(velocity_mesh, axis=0))
    pressure_mesh = torch.from_numpy(np.stack(pressure_mesh, axis=0))
    return velocity_mesh, pressure_mesh
