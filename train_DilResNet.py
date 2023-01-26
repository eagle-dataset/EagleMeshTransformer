import os
import torch
import torch.nn as nn
from Dataloader.IMG_Eagle import EagleDataset, grid2mesh
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.DilResNet import DilResNet
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--noise_std', default=0, type=float)
parser.add_argument('--horizon_train', default=6, type=int)
parser.add_argument('--horizon_val', default=25, type=int)
parser.add_argument('--batchsize', default=10, type=int)
parser.add_argument('--n_block', default=20, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--dataset_path', default='', type=str)
parser.add_argument('--name', default='', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MSE = nn.MSELoss()


def evaluate():
    print(args)
    length = 400
    dataset = EagleDataset(args.dataset_path, mode="test", window_length=length, with_mesh=True)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = DilResNet(noise_std=args.noise_std,
                      channels=4 if args.dataset == "fluent" else 3,
                      N_block=args.n_block).to(device)
    model.load_state_dict(torch.load(f"../trained_models/DRN/{args.name}.nn", map_location=device))

    with torch.no_grad():
        model.eval()
        error_velocity = torch.zeros(length - 1).to(device)
        error_pressure = torch.zeros(length - 1).to(device)

        os.makedirs(f"../Results/drn", exist_ok=True)
        for i, x in enumerate(tqdm(dataloader, desc="Evaluation")):
            states = x["states"].to(device).float()
            mask = x["mask"].to(device).float()

            state = states.permute(0, 1, 4, 2, 3)

            state_hat, _, _ = model(state, mask)
            state_hat = state_hat.permute(0, 1, 3, 4, 2)
            state_hat = dataset.denormalize(state_hat)

            state_hat = state_hat[0]
            velocity_grid = state_hat[..., :2].cpu().numpy()
            pressure_grid = state_hat[..., 2:].cpu().numpy()
            velocity_mesh, pressure_mesh = grid2mesh(velocity_grid, pressure_grid, x["mesh_pos"][0])

            velocity = x['mesh_velocity'][:, 1:]
            pressure = x['mesh_pressure'][:, 1:]
            velocity_hat = velocity_mesh[1:].unsqueeze(0)
            pressure_hat = pressure_mesh[1:].unsqueeze(0)

            mask = torch.ones_like(x["mesh_pos"])[..., 0]
            mask = mask[:, 1:].unsqueeze(-1)

            rmse_velocity = torch.sqrt((velocity[0] * mask[0] - velocity_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(
                1).to(device)
            rmse_pressure = torch.sqrt((pressure[0] * mask[0] - pressure_hat[0] * mask[0]).pow(2).mean(dim=-1)).mean(
                1).to(device)

            rmse_velocity = torch.cumsum(rmse_velocity, dim=0) / torch.arange(1, rmse_velocity.shape[0] + 1,
                                                                              device=device)
            rmse_pressure = torch.cumsum(rmse_pressure, dim=0) / torch.arange(1, rmse_pressure.shape[0] + 1,
                                                                              device=device)

            error_velocity = error_velocity + rmse_velocity
            error_pressure = error_pressure + rmse_pressure

        error_velocity = error_velocity / len(dataloader)
        error_pressure = error_pressure / len(dataloader)

        np.savetxt(f"../Results/drn/{args.dataset}/{args.name}_error_velocity.csv",
                   error_velocity.cpu().numpy(), delimiter=",")
        np.savetxt(f"../Results/drn/{args.dataset}/{args.name}_error_pressure.csv",
                   error_pressure.cpu().numpy(), delimiter=",")


def get_loss(state, state_hat, target, delta_hat):
    velocity = state[:, 1:, ..., :2]
    pressure = state[:, 1:, ..., 2:]
    velocity_hat = state_hat[:, 1:, ..., :2]
    pressure_hat = state_hat[:, 1:, ..., 2:]

    losses = {}

    loss = MSE(target, delta_hat)

    rmse_velocity = torch.sqrt(((velocity - velocity_hat) ** 2).mean(dim=(-1)))
    loss_velocity = torch.mean(rmse_velocity)
    rmse_pressure = torch.sqrt(((pressure - pressure_hat) ** 2).mean(dim=(-1)))
    loss_pressure = torch.mean(rmse_pressure)

    losses['MSE_pressure'] = loss_pressure
    losses['loss'] = loss
    losses['MSE_velocity'] = loss_velocity
    return losses


def validate(model, dataloader, epoch=0):
    with torch.no_grad():
        model.eval()
        loss_total, cpt = 0, 0
        for i, x in enumerate(tqdm(dataloader, desc="Validation")):
            states = x["states"].to(device).float()
            mask = x["mask"].to(device).float()

            state = states.permute(0, 1, 4, 2, 3)
            state_hat, target, delta_hat = model(state, mask, apply_noise=False)

            state = state.permute(0, 1, 3, 4, 2)
            state_hat = state_hat.permute(0, 1, 3, 4, 2)

            state = dataloader.dataset.denormalize(state)
            state_hat = dataloader.dataset.denormalize(state_hat)

            costs = get_loss(state, state_hat, target, delta_hat)
        loss_total += costs['loss'].item()
        cpt += states.shape[0]

    results = loss_total / cpt
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def main():
    print(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    name = args.name
    train_dataset = EagleDataset(args.dataset_path, mode="train", window_length=args.horizon_train)
    valid_dataset = EagleDataset(args.dataset_path, mode="valid", window_length=args.horizon_val)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4,
                                  pin_memory=True)

    model = DilResNet(noise_std=args.noise_std,
                      channels=4,
                      N_block=args.n_block).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    memory = torch.inf
    for epoch in range(args.epoch):
        model.train()
        for i, x in enumerate(tqdm(train_dataloader, desc="Training")):
            states = x["states"].to(device).float()
            mask = x["mask"].to(device).float()

            state = states.permute(0, 1, 4, 2, 3)
            state_hat, target, delta_hat = model(state, mask)

            state = state.permute(0, 1, 3, 4, 2)
            state_hat = state_hat.permute(0, 1, 3, 4, 2)

            state = train_dataset.denormalize(state)
            state_hat = train_dataset.denormalize(state_hat)

            costs = get_loss(state, state_hat, target, delta_hat)

            optim.zero_grad()
            costs['loss'].backward()
            optim.step()

        error = validate(model, valid_dataloader, epoch=epoch)
        if error < memory:
            memory = error
            os.makedirs(f"../trained_models/DRN", exist_ok=True)
            torch.save(model.state_dict(), f"../trained_models/DRN/{name}.nn")
            print("Saved!")
    validate(model, valid_dataloader)


if __name__ == '__main__':
    if args.epoch == 0:
        evaluate()
    else:
        main()
