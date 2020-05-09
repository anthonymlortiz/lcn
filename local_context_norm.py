import torch
from torch import nn
from torch.nn import functional as F
import math


class LocalContextNorm(nn.Module):
    def __init__(self, num_features, channels_per_group=2, window_size=(227, 227), eps=1e-5):
        super(LocalContextNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.window_size = window_size

    def forward(self, x):
        N, C, H, W = x.size()
        G = C // self.channels_per_group
        assert C % self.channels_per_group == 0
        if self.window_size[0] < H and self.window_size[1] < W:
            # Build integral image
            device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
            x_squared = x * x
            integral_img = x.cumsum(dim=2).cumsum(dim=3)
            integral_img_sq = x_squared.cumsum(dim=2).cumsum(dim=3)
            # Dilation
            d = (1, self.window_size[0], self.window_size[1])
            integral_img = torch.unsqueeze(integral_img, dim=1)
            integral_img_sq = torch.unsqueeze(integral_img_sq, dim=1)
            kernel = torch.tensor([[[[[1., -1.], [-1., 1.]]]]]).to(device)
            c_kernel = torch.ones((1, 1, self.channels_per_group, 1, 1)).to(device)
            with torch.no_grad():
                # Dilated conv
                sums = F.conv3d(integral_img, kernel, stride=[1, 1, 1], dilation=d)
                sums = F.conv3d(sums, c_kernel, stride=[self.channels_per_group, 1, 1])
                squares = F.conv3d(integral_img_sq, kernel, stride=[1, 1, 1], dilation=d)
                squares = F.conv3d(squares, c_kernel, stride=[self.channels_per_group, 1, 1])
            n = self.window_size[0] * self.window_size[1] * self.channels_per_group
            means = torch.squeeze(sums / n, dim=1)
            var = torch.squeeze((1.0 / n * (squares - sums * sums / n)), dim=1)
            _, _, h, w = means.size()
            pad2d = (int(math.floor((W - w) / 2)), int(math.ceil((W - w) / 2)), int(math.floor((H - h) / 2)),
                     int(math.ceil((H - h) / 2)))
            padded_means = F.pad(means, pad2d, 'replicate')
            padded_vars = F.pad(var, pad2d, 'replicate') + self.eps
            for i in range(G):
                x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] = \
                    (x[:, i * self.channels_per_group:i * self.channels_per_group + self.channels_per_group, :, :] -
                     torch.unsqueeze(padded_means[:, i, :, :], dim=1).to(device)) /\
                    ((torch.unsqueeze(padded_vars[:, i, :, :], dim=1)).to(device)).sqrt()
            del integral_img
            del integral_img_sq
        else:
            x = x.view(N, G, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C, H, W)

        return x * self.weight + self.bias
