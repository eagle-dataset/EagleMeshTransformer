import torch
import torch.nn as nn
from Models.Base import Normalizer

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class DilResNet(nn.Module):
    def __init__(self, channels=3, N_block=4, noise_std=0.0):
        """ Dilated Residual Network
        :param channels: number of channels in the input
        :param N_block: number of blocks in the network
        :param noise_std: standard deviation of the noise added to the input"""
        super(DilResNet, self).__init__()
        self.encoder = nn.Conv2d(channels, 48, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([CNNBlock() for _ in range(N_block)])
        self.decoder = nn.Conv2d(48, channels, kernel_size=3, padding=1)

        self.noise_std = noise_std

    def forward(self, state, mask, apply_noise=True):

        if apply_noise:
            noise_mask = torch.logical_or(mask[:, 0] == NODE_NORMAL, mask[:, 0] == NODE_OUTPUT)
            noise_mask = noise_mask.unsqueeze(1).repeat(1, state.shape[2], 1, 1)
            noise = torch.randn_like(state[:, 0]).to(state.device) * self.noise_std
            state[:, 0] = state[:, 0] + noise * noise_mask

        state_hat, delta = [state[:, 0]], []
        target = []

        for i in range(1, state.shape[1]):
            x = state_hat[-1]

            y = self.encoder(x)
            for block in self.blocks:
                y = y + block(y)
            y = self.decoder(y)

            delta.append(y)
            next_state = state_hat[-1] + y

            target.append(state[:, i] - state_hat[-1])

            boundaries = torch.logical_or(mask[:, i] == NODE_WALL, mask[:, i] == NODE_INPUT)
            boundaries = torch.logical_or(boundaries, mask[:, i] == NODE_DISABLE)
            boundaries = boundaries.unsqueeze(1).repeat(1, next_state.shape[1], 1, 1)

            next_state[boundaries] = state[:, i][boundaries]
            state_hat.append(next_state)

        delta = torch.stack(delta, dim=1)
        state_hat = torch.stack(state_hat, dim=1)
        target = torch.stack(target, dim=1)
        return state_hat, delta, target


class CNNBlock(nn.Module):
    def __init__(self):
        # Dilated CNN Layer
        super(CNNBlock, self).__init__()
        # 7 convolution layer with kernel size 3x3 and stride 1 and output channel 48
        self.conv1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=4, dilation=4)
        self.conv6 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=2, dilation=2)
        self.conv7 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1, dilation=1)

        self.f = nn.Sequential(self.conv1, nn.ReLU(),
                               self.conv2, nn.ReLU(),
                               self.conv3, nn.ReLU(),
                               self.conv4, nn.ReLU(),
                               self.conv5, nn.ReLU(),
                               self.conv6, nn.ReLU(),
                               self.conv7, nn.ReLU())

    def forward(self, x):
        return self.f(x)


if __name__ == '__main__':
    model = DilResNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
