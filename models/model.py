import torch
from torch import nn
from models.cnn import CNN_model


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNet(nn.Module):
    def __init__(self, yf,lambda0):
        super(DCFNet, self).__init__()
        self.feature = CNN_model()
        self.yf = yf.clone()
        self.lambda0 = lambda0

    def forward(self, z, x):  #size of x and z = [3,1,250,250]
        z = self.feature(z)
        x = self.feature(x)
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)

        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
        alphaf = self.yf.to(device=z.device) / (kzzf + self.lambda0)  # very Ugly
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
        return response
