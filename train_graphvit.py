"""Train script for GraphViT model on the Eagle dataset.

Args:
    epoch (int): number of epochs to train the model. Default is 1000
    lr (float): learning rate for the optimizer. Default is 1e-4
    data_path (Path): path to the dataset.
    n_cluster (int): number of clusters to use. Use 1, 10, 20, 30 or 40
    horizon_val (int): number of timestep to validate on. Default is 25
    horizon_train (int): number of timestep to train on. Default is 6
    w_size (int): size of the latent representation. Default is 512
    alpha (float): Weighting for the pressure term in the loss. Default is 0.1
    batch_size (int): batch size for training. Default is 2

Notes:
    Contrary to what mentioned in the appendix of the paper, the model is trained with
    a training window of 6 time steps (and not 8).
"""

import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from Models.graphViT import GraphViT
from Dataloader.eagle import EagleDataset, collate

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=Path)
parser.add_argument("--cluster-path", type=Path)
parser.add_argument("--run-id", type=str)
parser.add_argument("--output-path", type=Path)
parser.add_argument("--n-cluster", type=int)
parser.add_argument("--epoch", default=1000, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--horizon-val", default=25, type=int)
parser.add_argument("--horizon-train", default=6, type=int)
parser.add_argument("--w-size", default=512, type=int)
parser.add_argument("--alpha", default=0.1, type=float)
parser.add_argument("--batch-size", default=2, type=int)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()


def get_loss(velocity, pressure, output, state_hat, target, mask):
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :, :2]
    mask = mask[:, 1:].unsqueeze(-1)

    rmse_velocity = torch.sqrt(
        ((velocity * mask - velocity_hat * mask) ** 2).mean(dim=(-1))
    )
    loss_velocity = torch.mean(rmse_velocity)
    losses = {}

    pressure_hat = state_hat[:, 1:, :, 2:]
    rmse_pressure = torch.sqrt(
        ((pressure * mask - pressure_hat * mask) ** 2).mean(dim=(-1))
    )
    loss_pressure = torch.mean(rmse_pressure)
    loss = MSE(target[..., :2] * mask, output[..., :2] * mask) + args.alpha * MSE(
        target[..., 2:] * mask, output[..., 2:] * mask
    )
    loss = loss

    losses["MSE_pressure"] = loss_pressure
    losses["loss"] = loss
    losses["MSE_velocity"] = loss_velocity

    return losses


def validate(model: nn.Module, dataloader: DataLoader, epoch: int = 0):
    with torch.no_grad():
        total_loss, cpt = 0, 0
        model.eval()
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
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

            state_hat[..., :2], state_hat[..., 2:] = dataloader.dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = dataloader.dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)
            total_loss += costs["loss"].item()
            cpt += mesh_pos.shape[0]
    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    train_dataset = EagleDataset(
        args.dataset_path,
        cluster_path=args.cluster_path,
        split="train",
        window_length=args.horizon_train,
        n_cluster=args.n_cluster,
        normalized=True,
    )
    valid_dataset = EagleDataset(
        args.dataset_path,
        cluster_path=args.cluster_path,
        split="valid",
        window_length=args.horizon_val,
        n_cluster=args.n_cluster,
        normalized=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    model = GraphViT(state_size=4, w_size=args.w_size).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    memory = torch.inf
    output_ckpt = (args.output_path / args.run_id).with_suffix(".nn")
    output_ckpt.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epoch):
        model.train()

        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
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
                apply_noise=True,
            )

            state_hat[..., :2], state_hat[..., 2:] = train_dataset.denormalize(
                state_hat[..., :2], state_hat[..., 2:]
            )
            velocity, pressure = train_dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)

            optim.zero_grad()
            costs["loss"].backward()
            optim.step()

        error = validate(model, valid_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            torch.save(model.state_dict(), output_ckpt)
            print("Saved!")
    validate(model, valid_dataloader)


if __name__ == "__main__":
    main()
