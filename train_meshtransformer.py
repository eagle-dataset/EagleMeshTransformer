import torch
import torch.nn as nn
from Dataloader.eagle import EagleDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.MeshTransformer import MeshTransformer
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1000, type=int, help="Number of epochs, set to 0 to evaluate")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--dataset_path', default="", type=str,
                    help="Dataset path, caution, the cluster location is induced from this path, make sure this is Ok")
parser.add_argument('--horizon_val', default=25, type=int, help="Number of timestep to validate on")
parser.add_argument('--horizon_train', default=6, type=int, help="Number of timestep to train on")
parser.add_argument('--n_cluster', default=20, type=int, help="Number of nodes per cluster. 0 means no clustering")
parser.add_argument('--w_size', default=512, type=int, help="Dimension of the latent representation of a cluster")
parser.add_argument('--alpha', default=0.1, type=float, help="Weighting for the pressure term in the loss")
parser.add_argument('--batchsize', default=1, type=int, help="Batch size")
parser.add_argument('--name', default='', type=str, help="Name for saving/loading weights")
args = parser.parse_args()

BATCHSIZE = args.batchsize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()


def evaluate():
    print(args)
    length = 400
    dataset = EagleDataset(args.dataset_path, mode="test", window_length=length,
                           with_cluster=True, n_cluster=args.n_cluster, normalize=True, with_cells=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,
                            collate_fn=collate)
    model = MeshTransformer(state_size=4, w_size=args.w_size).to(device)

    model.load_state_dict(
        torch.load(f"../trained_models/graphvit/{args.name}.nn", map_location=device))

    with torch.no_grad():
        model.eval()

        error_velocity = torch.zeros(length - 1).to(device)
        error_pressure = torch.zeros(length - 1).to(device)

        os.makedirs(f"../Results/graphvit", exist_ok=True)
        for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type, clusters, clusters_mask,
                                              apply_noise=False)
            state_hat[..., :2], state_hat[..., 2:] = dataset.denormalize(state_hat[..., :2], state_hat[..., 2:])

            velocity, pressure = dataset.denormalize(velocity, pressure)

            velocity = velocity[:, 1:]
            pressure = pressure[:, 1:]
            velocity_hat = state_hat[:, 1:, :, :2]
            pressure_hat = state_hat[:, 1:, :, 2:]
            mask = mask[:, 1:].unsqueeze(-1)

            rmse_velocity = torch.sqrt((velocity[0] * mask[0] - velocity_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(1)
            rmse_pressure = torch.sqrt((pressure[0] * mask[0] - pressure_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(1)

            rmse_velocity = torch.cumsum(rmse_velocity, dim=0) / torch.arange(1, rmse_velocity.shape[0] + 1,
                                                                              device=device)
            rmse_pressure = torch.cumsum(rmse_pressure, dim=0) / torch.arange(1, rmse_pressure.shape[0] + 1,
                                                                              device=device)

            error_velocity = error_velocity + rmse_velocity
            error_pressure = error_pressure + rmse_pressure

    error_velocity = error_velocity / len(dataloader)
    error_pressure = error_pressure / len(dataloader)

    np.savetxt(f"../Results/graphvit/{args.name}_error_velocity.csv", error_velocity.cpu().numpy(), delimiter=",")
    np.savetxt(f"../Results/graphvit/{args.name}_error_pressure.csv", error_pressure.cpu().numpy(), delimiter=",")


def collate(X):
    """ Convoluted function to stack simulations together in a batch. Basically, we add ghost nodes
    and ghost edges so that each sim has the same dim. This is useless when batchsize=1 though..."""
    N_max = max([x["mesh_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])
    C_max = max([x["cluster"].shape[-2] for x in X])

    for batch, x in enumerate(X):
        # This step add fantom nodes to reach N_max + 1 nodes
        for key in ['mesh_pos', 'velocity', 'pressure']:
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
                [x["cluster"], N_max * torch.ones(x["cluster"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)
            x["cluster_mask"] = torch.cat(
                [x["cluster_mask"], torch.zeros(x["cluster_mask"].shape[0], C_max - c, x['cluster'].shape[-1])], dim=1)

        edges = x['edges']
        T, E, S = edges.shape
        x['edges'] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)

        x['mask'] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)

    output = {key: None for key in X[0].keys()}
    for key in output.keys():
        if key != "example":
            output[key] = torch.stack([x[key] for x in X], dim=0)
        else:
            output[key] = [x[key] for x in X]

    return output


def get_loss(velocity, pressure, output, state_hat, target, mask):
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :, :2]
    mask = mask[:, 1:].unsqueeze(-1)

    rmse_velocity = torch.sqrt(((velocity * mask - velocity_hat * mask) ** 2).mean(dim=(-1)))
    loss_velocity = torch.mean(rmse_velocity)
    losses = {}

    pressure_hat = state_hat[:, 1:, :, 2:]
    rmse_pressure = torch.sqrt(((pressure * mask - pressure_hat * mask) ** 2).mean(dim=(-1)))
    loss_pressure = torch.mean(rmse_pressure)
    loss = MSE(target[..., :2] * mask, output[..., :2] * mask) + args.alpha * MSE(target[..., 2:] * mask,
                                                                                  output[..., 2:] * mask)
    loss = loss

    losses['MSE_pressure'] = loss_pressure
    losses['loss'] = loss
    losses['MSE_velocity'] = loss_velocity

    return losses


def validate(model, dataloader, epoch=0, vizu=False):
    with torch.no_grad():
        total_loss, cpt = 0, 0
        model.eval()
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type, clusters, clusters_mask,
                                              apply_noise=False)

            state_hat[..., :2], state_hat[..., 2:] = dataloader.dataset.denormalize(state_hat[..., :2],
                                                                                    state_hat[..., 2:])
            velocity, pressure = dataloader.dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)
            total_loss += costs['loss'].item()
            cpt += mesh_pos.shape[0]
    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    name = args.name

    train_dataset = EagleDataset(args.dataset_path, mode="train", window_length=args.horizon_train, with_cluster=True,
                                 n_cluster=args.n_cluster, normalize=True)
    valid_dataset = EagleDataset(args.dataset_path, mode="valid", window_length=args.horizon_val, with_cluster=True,
                                 n_cluster=args.n_cluster, normalize=True)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4,
                                  pin_memory=False, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4,
                                  pin_memory=True, collate_fn=collate)

    model = MeshTransformer(state_size=4, w_size=args.w_size).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    memory = torch.inf
    for epoch in range(args.epoch):
        model.train()

        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = x["mask"].to(device)
            clusters = x["cluster"].to(device).long()
            clusters_mask = x["cluster_mask"].to(device).long()

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type, clusters, clusters_mask,
                                              apply_noise=True)

            state_hat[..., :2], state_hat[..., 2:] = train_dataset.denormalize(state_hat[..., :2],
                                                                               state_hat[..., 2:])
            velocity, pressure = train_dataset.denormalize(velocity, pressure)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)

            optim.zero_grad()
            costs['loss'].backward()
            optim.step()

        error = validate(model, valid_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            os.makedirs(f"../trained_models/graphvit/", exist_ok=True)
            torch.save(model.state_dict(), f"../trained_models/graphvit/{name}.nn")
            print("Saved!")
    validate(model, valid_dataloader)


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
