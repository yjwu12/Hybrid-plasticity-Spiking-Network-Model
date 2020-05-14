import torch
import torch.nn as nn
import torch.nn.functional as F
# testing batch_method, but further comparison should be going on tomorrow! to keep the coherence with above situation
import os
import numpy as np
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def kernel_size(cfg):
    dims = 32
    cfg_ = list([])
    for layer in cfg:
        dims_ = (dims - layer[4] + 1 + 2 * layer[3]) / layer[2]
        cfg_.append(int(dims_))
        dims = dims_
    return cfg_


thresh = 0.5
learning_rate = 0.08
lens = 0.5
probs = 0.5
decay = 0.7
tau_w = 50
num_classes = 10
batch_size = 50
num_epochs = 120
num_update = 1 # using a larger number of updates can achieve more stable learning process
time_window = 8 # using T = 8-10 for better results , default T = 8
act_fun = ActFun.apply

# in_plane,out_plane, stride,padding, kernel_size
cfg_cnn = [
    (3, 128, 1, 1, 3),

    (128, 256, 1, 1, 3),

    (256, 512, 1, 1, 3),

    (512, 1024, 1, 1, 3),

    (1024, 512, 1, 1, 3),

]

cfg_fc = [1024, 512, 10]

cnn_dim = [32, 32, 16, 8, 8]



def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
        print('successfully reset lr')
    return optimizer


fc_dim = int(cfg_cnn[-1][1] * cnn_dim[-1] * cnn_dim[-1])


class SCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[3]
        self.conv4 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[4]
        self.conv5 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)



        self.fc1 = nn.Linear(fc_dim, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

        self.shift = [np.power(1.1, x) for x in range(9)]
        # in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[5]
        # self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(fc_dim, cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2])

        self.conv2.weight.data = (self.conv2.weight.data + self.conv2.weight.data.mean() * self.shift[2])
        self.conv3.weight.data = (self.conv3.weight.data + self.conv3.weight.data.mean() * self.shift[3])
        self.conv4.weight.data = (self.conv4.weight.data + self.conv4.weight.data.mean() * self.shift[4])
        self.conv5.weight.data = (self.conv5.weight.data + self.conv5.weight.data.mean() * self.shift[5])
        self.fc1.weight.data = (self.fc1.weight.data + self.fc1.weight.data.mean() * self.shift[6])
        self.fc2.weight.data = (self.fc2.weight.data + self.fc2.weight.data.mean() * self.shift[7])
        # self.fc3.weight.data = self.fc1.weight.data

        self.alpha1 = torch.nn.Parameter((1e-3 * torch.rand(1)).cuda(), requires_grad=True)

        self.alpha2 = torch.nn.Parameter((1e-3 * torch.rand(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((1e-3 * torch.ones(1)).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((1e-3 * torch.ones(1)).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-3 * torch.rand(1, fc_dim)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)

    def produce_hebb(self):
        hebb1 = torch.zeros(fc_dim, cfg_fc[0], device=device).float()
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device).float()
        return hebb1, hebb2

    def parameter_split(self):
        base_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'co':
                base_param.append(p)
                print(n)

        local_param = list(set(self.parameters()) - set(base_param))
        return base_param, local_param

    def forward(self, input, hebb1, hebb2, coding ='rank'):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cnn_dim[0], cnn_dim[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cnn_dim[1], cnn_dim[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cnn_dim[2], cnn_dim[2], device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], cnn_dim[3], cnn_dim[3], device=device)
        c5_mem = c5_spike = torch.zeros(batch_size, cfg_cnn[4][1], cnn_dim[4], cnn_dim[4], device=device)
        # c6_mem = c6_spike = torch.zeros(batch_size, cfg_cnn[5][1], cnn_dim[5], cnn_dim[5], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)


        for step in range(time_window):
            # x = input > torch.rand(input.size(), device=device)
            k_filter = math.exp(- step / tau_w)

            x = input

            c1_mem, c1_spike = mem_update_nonplastic(self.conv1, x.float(), c1_spike, c1_mem )

            c2_mem, c2_spike = mem_update_nonplastic(self.conv2,
                                                    F.dropout(c1_spike * k_filter, p=probs, training=self.training),
                                                    c2_spike, c2_mem )

            x = F.avg_pool2d(c2_spike, 2)

            c3_mem, c3_spike = mem_update_nonplastic(self.conv3,
                                                    F.dropout(x * k_filter, p=probs, training=self.training), c3_spike,
                                                    c3_mem )

            x = F.avg_pool2d(c3_spike, 2)

            c4_mem, c4_spike = mem_update_nonplastic(self.conv4,
                                                    F.dropout(x * k_filter, p=probs, training=self.training), c4_spike,
                                                    c4_mem )
            #
            c5_mem, c5_spike = mem_update_nonplastic(self.conv5,
                                                    F.dropout(c4_spike * k_filter, p=probs, training=self.training),
                                                    c5_spike, c5_mem)

            x = c5_spike.view(batch_size, -1)

            # OPTIONS: you can run the following code for higher accuracy
            # h1_mem, h1_spike, hebb1 = mem_update_plastic(self.fc1, self.alpha1, self.beta1, self.gamma1, self.eta1 x,
            #                                      h1_spike, h1_mem, hebb1)

            h1_mem, h1_spike = mem_update_nonplastic(self.fc1, F.dropout(x * k_filter, p=probs, training=self.training),
                                                  h1_spike, h1_mem )

            h2_mem, h2_spike, hebb2 = mem_update_plastic(self.fc2, self.alpha2, self.beta2, self.gamma2, self.eta2, h1_spike,
                                                 h2_spike, h2_mem, hebb2)

            h3_mem = h3_mem * decay + self.fc3(h2_spike * k_filter)

            buf = h3_mem.gt(thresh).float().max(dim=1)[0].detach_().mean()
            if buf > 0.7 and step > 0: break
            # coding selection
            # if coding == 'rank':
            #     outs = h3_mem.clamp(max=thresh * 1.1) / thresh
            # elif coding == 'rate:
            #     outs = h3_sumspike / time_window

        outs = h3_mem.clamp(max=thresh * 1.1) / thresh

        return outs, step, hebb1.detach(), hebb2.detach(), self.eta1, self.eta2


# test()

def mem_update_nonplastic(fc, x, spike, mem ):
    state = fc(x)
    mem = (mem - thresh * spike) * decay + state
    now_spike = act_fun(mem - thresh)
    return mem, now_spike.float()


def mem_update_plastic(fc, alpha, beta, gamma, eta, inputs, spike, mem, hebb):
    state = fc(inputs) + alpha * inputs.mm(hebb)
    mem = (mem - thresh * spike) * decay + state
    now_spike = act_fun(mem - thresh)
    hebb = 0.9  * hebb - torch.bmm((inputs * beta).unsqueeze(2), ((mem / thresh) - eta).unsqueeze(1)).mean(dim=0).squeeze()
    return mem, now_spike.float(), hebb