import torch.nn.functional as F
from torch.nn.parameter import Parameter

class ruihua(nn.Module):
    def __init__(self, padding=2):
        super(ruihua, self).__init__()
        kernel = [[-1, -1, -1, -1, -1],
                  [-1, 2, 2, 2, -1],
                  [-1, 2, 8, 2, -1],
                  [-1, 2, 2, 2, -1],
                  [-1, -1, -1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)

        x = torch.cat([x1, x2], dim=1)
        return x

class ssa(nn.Module):
    def __init__(self, k_size=1):
        super(ssa, self).__init__()
        self.conv11 = nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.conv33 = nn.Conv2d(4, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.rh = ruihua(2)

    def forward(self, x):
        module = x
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        # x = torch.cat([x, self.rh(x)], dim=1)
        x = self.rh(x)
        x = self.conv11(x)
        x = self.sigmoid(x)
        return module * x