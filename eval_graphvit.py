"""Evaluate the GraphViT model on the Eagle dataset.

Args:
    --ckpt: path to the model checkpoint
    --dataset_path: path to the dataset
    --cluster_path: path to the precomputed clusters
    --n_cluster: number of clusters to use. Use 1, 10, 20, 30 or 40
    --w_size: size of the latent representation. Default is 512
"""

import torch
import random
import argparse

import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path

from Models.graphViT import GraphViT
from torch.utils.data import DataLoader
from Dataloader.eagle import EagleDataset, collate

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=Path)
parser.add_argument("--dataset-path", type=Path)
parser.add_argument("--cluster-path", type=Path)
parser.add_argument("--n-cluster", default=20, type=int)
parser.add_argument("--w-size", default=512, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()


def evaluate()->None:
    """Compute the N-RMSE on the test set of Eagle.

    Predictions are sequences of shape (T, N, 2) (for both velocity and pressure). After
    denormalization, we compute the RMSE for each time step over the last dimension,
    and take the mean over the nodes. We then cumulate the RMSE over time and divide by
    the time step to get the metric at each time step. Finally, we normalize the error
    by the (norm of) the standard deviation of the velocity and pressure.
    """
    torch.manual_seed(3721)
    torch.cuda.manual_seed(3721)
    np.random.seed(3721)
    random.seed(3721)

    L = 251
    dataset = EagleDataset(
        data_path=args.dataset_path,
        cluster_path=args.cluster_path,
        split="test",
        window_length=L,
        n_cluster=args.n_cluster,
        normalized=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )
    model = GraphViT(state_size=4, w_size=args.w_size).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))

    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        model.eval()
        arange = torch.arange(1, L).to(device)
        error_velocity = torch.zeros(L - 1).to(device)
        error_pressure = torch.zeros(L - 1).to(device)

        p_std = torch.sqrt((dataset.pressure_std**2).sum(-1)).to(device)
        v_std = torch.sqrt((dataset.velocity_std**2).sum(-1)).to(device)

        for i, x in enumerate(dataloader):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x["edges"].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(
                mesh_pos,
                edges,
                state,
                node_type,
                clusters,
                clusters_mask,
                apply_noise=False,
            )

            # Denormalize
            velocity_hat, pressure_hat = state_hat[..., :2], state_hat[..., 2:]
            velocity_hat, pressure_hat = dataset.denormalize(velocity_hat, pressure_hat)
            velocity, pressure = dataset.denormalize(velocity, pressure)

            # Remove the initial state
            v_gt, p_gt = velocity[0, 1:], pressure[0, 1:]
            v_hat, p_hat = velocity_hat[0, 1:], pressure_hat[0, 1:]
            mask = mask[0, 1:].unsqueeze(-1)

            # Error Norm
            rmse_velocity = torch.sqrt(((v_gt * mask - v_hat * mask) ** 2).sum(dim=-1))
            rmse_pressure = torch.sqrt(((p_gt * mask - p_hat * mask) ** 2).sum(dim=-1))

            # Mean over the nodes
            rmse_velocity = rmse_velocity.mean(1)
            rmse_pressure = rmse_pressure.mean(1)

            # Cumulative RMSE
            rmse_velocity = torch.cumsum(rmse_velocity, dim=0) / arange
            rmse_pressure = torch.cumsum(rmse_pressure, dim=0) / arange

            # Accumulate the error over the batch
            error_velocity = error_velocity + rmse_velocity
            error_pressure = error_pressure + rmse_pressure

            # Compute the normalized error
            error_v = error_velocity / (i + 1) / v_std
            error_p = error_pressure / (i + 1) / p_std
            error = error_p + error_v
            pbar.set_postfix(
                dict(
                    error_1=error[0].item(),
                    error_50=error[49].item(),
                    error_250=error[249].item(),
                )
            )
            pbar.update(1)


if __name__ == "__main__":
    evaluate()
