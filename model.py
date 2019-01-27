from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import constants


class SqueezeExcite(nn.Module):
    def __init__(self, input_filters: int, r: int = 8):
        super().__init__()
        self._r = r
        self.layers = nn.Sequential(
            nn.Linear(input_filters, input_filters // r),
            nn.ReLU(),
            nn.Linear(input_filters // r, input_filters),
            nn.Sigmoid()
        )

    def forward(self, x):
        inputs = x
        b, c, h, w = x.size()
        x = x.view(b, c, h * w).mean(dim=-1)
        x = self.layers(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return inputs * x


class Residual(nn.Module):
    def __init__(self, input_filters: int, output_filters: int, stride: int = 2, drop: float = 0.3, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(input_filters, output_filters, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_filters),
            nn.ReLU(),
            nn.Dropout2d(drop),
            nn.Conv2d(output_filters, output_filters, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(output_filters),
        )
        self.out_relu = nn.ReLU()
        self._stride = stride
        self._out_filters = output_filters
        if stride == 1 and input_filters == output_filters:
            self.identity = lambda x: x
        else:
            self.identity = nn.Sequential(
                nn.Conv2d(input_filters, output_filters, 1, stride=stride, padding=0),
                nn.BatchNorm2d(output_filters)
            )

        self.se_module = SqueezeExcite(output_filters, r=8)

    def forward(self, x):
        return self.out_relu(self.se_module(self.layers(x)) + self.identity(x))


class CapsLoss(nn.Module):
    def __init__(self, m_plus: float = 0.9, m_minus: float = 0.1, lmbd: float = 0.5, reconstruction_lmbd: float = 5e-4):
        super().__init__()
        self._m_plus = m_plus
        self._m_minus = m_minus

        self._lmbd = lmbd
        self._reconstruction_lmbd = reconstruction_lmbd

        self._mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, true, reconstruction, true_image):
        output_lengths = pred.norm(dim=-1)
        l_k = true * torch.pow(F.relu(self._m_plus - output_lengths), 2) + self._lmbd * (1 - true) * torch.pow(
            F.relu(output_lengths - self._m_minus), 2)
        l_reconstruction = self._mse_loss(reconstruction, true_image)
        total_loss = l_k.sum() + self._reconstruction_lmbd * l_reconstruction
        return total_loss / pred.size(0)


def squash(x, axis=-1):
    normed = x.norm(p=2, dim=axis, keepdim=True)
    squared_norm = torch.pow(normed, 2)
    squashed = squared_norm / (squared_norm + 1) * x / normed
    return squashed


class SpectoCapsLayer(nn.Module):
    def __init__(self, input_filters: int, output_filters: int, input_caps: int, output_caps: int,
                 routing_steps: int = 3):
        super().__init__()
        self._input_filters = input_filters
        self._output_filters = output_filters
        self._output_caps = output_caps
        self._routing_steps = routing_steps

        self._weights = nn.Parameter(torch.randn(input_caps, output_caps, input_filters, output_filters))

    def forward(self, x):
        u_dash = x[:, :, None, None, :] @ self._weights
        u_dash = torch.squeeze(u_dash, dim=3)
        logits = Variable(torch.zeros(u_dash.size(0), u_dash.size(1), u_dash.size(2), 1).float()).cuda()
        v_j = u_dash
        for _ in range(self._routing_steps):
            c_is = []
            for i in range(logits.size(2)):
                c_is.append(F.softmax(logits[:, :, i], dim=1))
            c_i = torch.stack(c_is, dim=2)
            s_j = torch.sum(c_i * u_dash, dim=1, keepdim=True)
            v_j = squash(s_j)
            logits = logits + torch.sum(u_dash * v_j, dim=-1, keepdim=True)
        return torch.squeeze(v_j, dim=1), u_dash


class FastPrimaryLayer(nn.Module):
    def __init__(self, input_filters: int, output_filters: int, caps_dimension: int, kernel_size: int = 9,
                 stride: int = 2):
        super().__init__()
        self._layers = nn.ModuleList([
            nn.Conv2d(input_filters, output_filters, kernel_size=3, stride=stride)
            for _ in range(caps_dimension)])

    def forward(self, x):
        output = [layer(x).view(x.size(0), -1, 1) for layer in self._layers]
        output = torch.cat(output, dim=-1)
        output = squash(output)
        return output


class Reconstruction(nn.Module):
    def __init__(self, spect_cap_size, num_classes):
        super().__init__()
        self._init = nn.Sequential(
            nn.ConvTranspose2d(spect_cap_size + num_classes, spect_cap_size + num_classes, 4, 2, padding=1),
            nn.ConvTranspose2d(spect_cap_size + num_classes, 1, 4, 2, padding=1),
        )
        self._final = nn.Linear(1024, 1024)

    def forward(self, x):
        result = self._init(x)
        b, c, h, w = result.size()
        result = self._final(result.view(b, - 1)).view(b, c, h, w)
        return result


class CapsNet(nn.Module):
    def __init__(self, num_classes: int = 3, primary_cap_size: int = 64, spect_cap_size: int = 16,
                 kernel_size: int = 9):
        super().__init__()
        self._kernel_size = kernel_size
        self._num_classes = num_classes
        self._bf = 32
        self._initial = nn.Sequential(
            Residual(1, self._bf, 2, 0.0, 3), Residual(self._bf, self._bf, 1, 0, 3),
            Residual(self._bf, self._bf, 2, 0.0, 3), Residual(self._bf, self._bf * 2, 1, 0, 3),
            Residual(self._bf * 2, self._bf * 2, 2, 0, 3), Residual(self._bf * 2, 64, 1, 0, 3),
        )

        self._output_size = (constants.INITIAL_SIZE - (kernel_size - 1) * 2) // 2
        self._primary_caps = FastPrimaryLayer(primary_cap_size, 1, primary_cap_size, kernel_size, 2)
        self._specto_caps = SpectoCapsLayer(primary_cap_size, spect_cap_size, self._output_size * self._output_size,
                                            num_classes)

        self._reconstruction = Reconstruction(spect_cap_size, num_classes)

    def forward(self, x, y: Optional[Variable] = None):
        x = self._initial(x)
        x = self._primary_caps(x)
        x, u_dash = self._specto_caps(x)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        if y is None:
            max_index = torch.argmax(classes, dim=-1)
            y = Variable(torch.eye(self._num_classes).float()).cuda().index_select(dim=0, index=max_index)

        one_hot_layers = Variable(
            torch.ones(x.size(0), self._num_classes, self._output_size, self._output_size).float()).cuda() * y[:, :,
                                                                                                             None, None]
        to_select = torch.argmax(y, dim=-1)
        mask = Variable(
            torch.ones(x.size(0), self._output_size * self._output_size, 1, u_dash.size(-1)).long()).cuda() * to_select[
                                                                                                              :, None,
                                                                                                              None,
                                                                                                              None]
        to_reconstruct = u_dash.gather(dim=2, index=mask).view(x.size(0), self._output_size, self._output_size,
                                                               u_dash.size(-1)).permute(0, 3, 1, 2)
        xaxa = torch.cat([one_hot_layers, to_reconstruct], dim=1)
        reconstruction = self._reconstruction(xaxa)

        return classes, x, reconstruction

    def get_num_parameters(self) -> int:
        return int(np.sum(np.prod(x.size()) for x in self.parameters()))


if __name__ == '__main__':
    model = CapsNet(3).cuda()
    random_data = Variable(torch.randn(8, 1, 32, 32)).cuda()
    res = model.forward(random_data)
    print(res[:2])
