import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
        x = x.mean(dim=-1)
        x = self.layers(x)
        x = x.view(x.size(0), x.size(1), 1)
        return inputs * x


class Residual(nn.Module):
    def __init__(self, input_filters: int, output_filters: int, stride: 2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_filters, output_filters, 3, stride=stride, padding=1),
            nn.BatchNorm1d(output_filters),
            nn.ReLU(),
            nn.Conv1d(output_filters, output_filters, 3, stride=1, padding=1),
            nn.BatchNorm1d(output_filters),
        )
        self.out_relu = nn.ReLU()
        self._stride = stride
        self._out_filters = output_filters
        if stride == 1 and input_filters == output_filters:
            self.identity = lambda x: x
        else:
            self.identity = nn.Sequential(
                nn.Conv1d(input_filters, output_filters, 1, stride=stride, padding=0),
                nn.BatchNorm1d(output_filters)
            )

        self.se_module = SqueezeExcite(output_filters, r=4)

    def forward(self, x):
        return self.out_relu(self.se_module(self.layers(x)) + self.identity(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            Residual(32, 128, 1),
            Residual(128, 256, 2),
            Residual(256, 256, 1),
            Residual(256, 256, 1)
        )

        self.lstm_1 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(128, 64)

        self.dense = nn.Linear(64, 3)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm_1(x, None)
        x, _ = self.lstm_2(x, None)
        x = self.dense(x[:, -1])
        classes = torch.argmax(x, dim=1)
        softmax = F.softmax(x, dim=1)
        return x, classes, softmax

    def get_num_parameters(self) -> int:
        return int(np.sum(np.prod(x.size()) for x in self.parameters()))


if __name__ == '__main__':
    model = Net().cuda()
    random_data = Variable(torch.randn(8, 32, 32)).cuda()
    res = model.forward(random_data)
    print(res[:2])
    print(model.get_num_parameters())
