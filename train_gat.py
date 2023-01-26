import os
import torch
import torch.nn as nn
from Dataloader.eagle import EagleDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.GraphAttention import GraphAttention
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1000, type=int, help="Number of epochs, set to 0 to evaluate")
parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
parser.add_argument('--w_pressure', default=0.1, type=float, help="Weighting for the pressure term in the loss")
parser.add_argument('--heads', default=2, type=int, help="Number of heads of each GAT layer")
parser.add_argument('--n_processor', default=10, type=int, help="Number of chained GAT layers")
parser.add_argument('--horizon_val', default=25, type=int, help="Number of timestep to validate on")
parser.add_argument('--horizon_train', default=6, type=int, help="Number of timestep to train on")
parser.add_argument('--dataset_path', default="", type=str, help="Dataset location")
parser.add_argument('--name', default='gat', type=str, help="Name for saving/loading weights")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()
BATCHSIZE = 1


def evaluate():
    print(args)
    length = 400
    dataset = EagleDataset(args.dataset_path, mode="test", window_length=length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)
    model = GraphAttention(apply_noise=True, state_size=4, n_heads=args.heads, N=args.n_processor).to(device)

    model.load_state_dict(torch.load(f"trained_models/gat/{args.name}.nn", map_location=device))
    os.makedirs(f"../Results/gat", exist_ok=True)
    with torch.no_grad():
        model.eval()
        model.apply_noise = False

        error_velocity = torch.zeros(length - 1).to(device)
        error_pressure = torch.zeros(length - 1).to(device)

        for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = torch.ones_like(mesh_pos)[..., 0]

            state = torch.cat([velocity, pressure], dim=-1)

            state_hat, output, target = model(mesh_pos, edges, state, node_type)

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

            error_velocity = error_velocity + torch.FloatTensor(rmse_velocity).to(device)
            error_pressure = error_pressure + torch.FloatTensor(rmse_pressure).to(device)

        error_velocity = error_velocity / len(dataloader)
        error_pressure = error_pressure / len(dataloader)

        np.savetxt(f"../Results/gat/{args.dataset}/{args.name}_error_velocity.csv",
                   error_velocity.cpu().numpy(), delimiter=",")
        np.savetxt(f"../Results/gat/{args.dataset}/{args.name}_error_pressure.csv",
                   error_pressure.cpu().numpy(), delimiter=",")


def collate(X):
    N_max = max([x["mesh_pos"].shape[-2] for x in X])
    E_max = max([x["edges"].shape[-2] for x in X])
    for x in X:
        for key in ['mesh_pos', 'velocity', 'pressure', 'node_type']:
            tensor = x[key]
            T, N, S = tensor.shape
            x[key] = torch.cat([tensor, torch.zeros(T, N_max - N + 1, S)], dim=1)
        edges = x['edges']
        T, E, S = edges.shape
        x['edges'] = torch.cat([edges, N_max * torch.ones(T, E_max - E + 1, S)], dim=1)
        x['mask'] = torch.cat([torch.ones(T, N), torch.zeros(T, N_max - N + 1)], dim=1)
    output = {key: None for key in X[0].keys()}
    for key in output.keys():
        output[key] = torch.stack([x[key] for x in X], dim=0)
    return output


def get_loss(velocity, pressure, output, state_hat, target, mask):
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :, :2]
    mask = mask[:, 1:].unsqueeze(-1)

    rmse_velocity = torch.sqrt(((velocity * mask - velocity_hat * mask) ** 2).mean(dim=(-1)))
    loss_velocity = torch.mean(rmse_velocity)
    loss = MSE(target[..., :2] * mask, output[..., :2] * mask)
    losses = {}

    pressure_hat = state_hat[:, 1:, :, 2:]
    rmse_pressure = torch.sqrt(((pressure * mask - pressure_hat * mask) ** 2).mean(dim=(-1)))
    loss_pressure = torch.mean(rmse_pressure)
    loss = loss + args.w_pressure * MSE(target[..., 2:] * mask, output[..., 2:] * mask)

    losses['MSE_pressure'] = loss_pressure
    losses['loss'] = loss
    losses['MSE_velocity'] = loss_velocity
    return losses


def validate(model, dataloader, epoch=0, vizu=False):
    with torch.no_grad():
        model.eval()
        model.apply_noise = False
        total_loss, cpt = 0, 0
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            node_type = x["node_type"].to(device)
            pressure = x["pressure"].to(device)
            mask = torch.ones_like(mesh_pos)[..., 0]
            state = torch.cat([velocity, pressure], dim=-1)

            state_hat, output, target = model(mesh_pos, edges, state, node_type)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)
            total_loss += costs['loss'].item()
            cpt += state.shape[0]

        model.apply_noise = True
    results = total_loss / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    batchsize = BATCHSIZE

    name = args.name

    train_dataset = EagleDataset(args.dataset_path, mode="train", window_length=args.horizon_train, with_cluster=False)
    valid_dataset = EagleDataset(args.dataset_path, mode="valid", window_length=args.horizon_val, with_cluster=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=1, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, num_workers=1, pin_memory=True)

    model = GraphAttention(apply_noise=True, state_size=4, n_heads=args.heads, N=args.n_processor).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.991)

    memory = torch.inf
    for epoch in range(args.epoch):
        model.train()
        model.apply_noise = True

        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            mesh_pos = x["mesh_pos"].to(device)
            edges = x['edges'].to(device).long()
            velocity = x["velocity"].to(device)
            pressure = x["pressure"].to(device)
            node_type = x["node_type"].to(device)
            mask = torch.ones_like(mesh_pos)[..., 0]

            state = torch.cat([velocity, pressure], dim=-1)
            state_hat, output, target = model(mesh_pos, edges, state, node_type)

            costs = get_loss(velocity, pressure, output, state_hat, target, mask)

            optim.zero_grad()
            costs['loss'].backward()
            optim.step()

        if scheduler.get_last_lr()[0] > 1e-6 and epoch > 1:
            scheduler.step()

        error = validate(model, valid_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            os.makedirs(f"../trained_models/gat/", exist_ok=True)
            torch.save(model.state_dict(), f"../trained_models/gat/{name}.nn")
            print("Saved!")

    validate(model, valid_dataloader)


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
