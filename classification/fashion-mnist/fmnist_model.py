import torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import math

thresh = 0.45
lens = 0.5
decay = 0.75
w_decay = 0.9
num_classes = 10
batch_size = 50
tau_w  = 30
num_epochs = 101
learning_rate = 5e-4
time_window = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np
import random
import matplotlib.pyplot as plt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()


cfg_cnn = [
    (1, 128, 1, 1, 3),

    (128, 256, 1, 1, 3),

    (256, 256, 1, 1, 3),

]

cnn_dim = [28, 14, 7, 3]

fc_dim = cfg_cnn[-1][1] * cnn_dim[-1] * cnn_dim[-1]

cfg_fc = [512, 10]

probs = 0.5
act_fun = ActFun.apply


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SNN_Model(nn.Module):

    def __init__(self, p=0.5):
        super(SNN_Model, self).__init__()

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(fc_dim, cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )

        self.fc2.weight.data = self.fc2.weight.data * 0.5
        self.mask1 = p > torch.rand(self.fc1.weight.size(), device=device)
        self.mask2 = p > torch.rand(self.fc2.weight.size(), device=device)
        self.mask1 = self.mask1.float()
        self.mask2 = self.mask2.float()

        self.alpha1 = torch.nn.Parameter((1e-1 * torch.ones(1)).cuda(), requires_grad=True)

        self.alpha2 = torch.nn.Parameter((1e-1 * torch.ones(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((.0 * torch.zeros(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((.0 * torch.zeros(1, cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((1e-2 * torch.ones(1)).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((1e-2 * torch.ones(1)).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-3 * torch.ones(1, fc_dim)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.ones(1, cfg_fc[0])).cuda(), requires_grad=True)

    def mask_weight(self):
        self.fc1.weight.data = self.fc1.weight.data * self.mask1
        self.fc2.weight.data = self.fc2.weight.data * self.mask2

    def produce_hebb(self):
        hebb1 = torch.zeros(fc_dim, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return hebb1, hebb2

    def forward(self, input, hebb1, hebb2, wins=time_window):
        batch_size = input.size(0)
        # self.mask_weight()

        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cnn_dim[0], cnn_dim[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cnn_dim[1], cnn_dim[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cnn_dim[2], cnn_dim[2], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        # hebb1 = torch.zeros(784, cfg_fc[0], device=device)

        for step in range(wins):
            decay_factor = np.exp(-step/tau_w)

            x = input

            c1_mem, c1_spike = mem_update_nonplastic(self.conv1, x.float(), c1_spike, c1_mem)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update_nonplastic(self.conv2, F.dropout(x * decay_factor, p=probs, training=self.training), c2_spike,
                                               c2_mem)
            x = F.avg_pool2d(c2_spike, 2)

            c3_mem, c3_spike = mem_update_nonplastic(self.conv3, F.dropout(x * decay_factor, p=probs, training=self.training), c3_spike,
                                               c3_mem)

            x = F.avg_pool2d(c3_spike, 2)

            x = x.view(batch_size, -1).float()

            h1_mem, h1_spike, hebb1 = mem_update(self.fc1, self.alpha1, self.beta1, self.gamma1, self.eta1, x * decay_factor, h1_spike,
                                                 h1_mem, hebb1)
            h1_sumspike = h1_sumspike + h1_spike

            h2_mem, h2_spike = mem_update_nonplastic(self.fc2, h1_spike * decay_factor, h2_spike, h2_mem)

            buf = h2_mem.gt(thresh).float().max(dim=1)[0].detach_().mean()
            if buf > 0.9 and step > 0:
                break

        outs = (h2_mem / thresh).clamp(max=1.1)

        return outs, h1_mem, hebb1.data, hebb2.data, self.eta1, self.eta2


#
def mem_update(fc, alpha, beta, gamma, eta, inputs, spike, mem, hebb):
    state = fc(inputs) + alpha * inputs.mm(hebb)
    mem = mem * (1 - spike) * decay + state
    now_spike = act_fun(mem - thresh)
    hebb = w_decay * hebb - torch.bmm((inputs * beta.clamp(min=0.)).unsqueeze(2),
                                   ((mem / thresh) - eta).unsqueeze(1)).mean(dim=0).squeeze()
    hebb = hebb.clamp(min=-4, max=4)
    return mem, now_spike.float(), hebb


def mem_update_nonplastic(fc, inputs, spike, mem):
    state = fc(inputs)
    mem = mem * (1 - spike) * decay + state
    now_spike = act_fun(mem - thresh)
    return mem, now_spike.float() 


