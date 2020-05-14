import torch, time, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

thresh = 0.3
lens = 0.5
mem_decay = 0.6
w_decay = 0.95
num_classes = 10
batch_size = 100
num_epochs = 100
learning_rate = 5e-4
time_window = 10
num_updates = 1
tau_w = 40

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def __init__(self):
        super(SNN_Model, self).__init__()

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(fc_dim, cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )


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

    def parameter_split(self):
        base_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'fv':
                base_param.append(p)

        local_param = list(set(self.parameters()) - set(base_param))
        return base_param, local_param

    def forward(self, input, hebb1, hebb2, wins=time_window, coding = 'rank'):

        c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, cfg_cnn[0][1], cnn_dim[0], cnn_dim[0], device=device)
        c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, cfg_cnn[1][1], cnn_dim[1], cnn_dim[1], device=device)
        c3_mem = c3_spike = c3_sumspike = torch.zeros(batch_size, cfg_cnn[2][1], cnn_dim[2], cnn_dim[2], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(wins):

            decay_factor = np.exp(- step/tau_w)

            x = input

            c1_mem, c1_spike = mem_update_nonplastic(self.conv1, x.float() , c1_spike, c1_mem)


            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update_nonplastic(self.conv2, F.dropout(x*decay_factor, p=probs, training=self.training),
                                               c2_spike, c2_mem)
            x = F.avg_pool2d(c2_spike, 2)

            c3_mem, c3_spike = mem_update_nonplastic(self.conv3, F.dropout(x*decay_factor, p=probs, training=self.training), c3_spike,
                                               c3_mem)

            x = F.avg_pool2d(c3_spike, 2)

            x = x.view(batch_size, -1).float()

            h1_mem, h1_spike, hebb1 = mem_update_plastic(self.fc1, self.alpha1, self.beta1, self.gamma1, self.eta1,
                                                 x * decay_factor, h1_spike, h1_mem, hebb1)

            h2_mem, h2_spike  = mem_update_nonplastic(self.fc2, h1_spike*decay_factor, h2_spike, h2_mem)

            h2_sumspike = h2_sumspike + h2_spike


            if coding == 'rank':
                buf = h2_mem.gt(thresh).float().max(dim=1)[0].detach_().mean()
                if buf > 0.8 and step > 0: break

        if coding == 'rank':
            outs = h2_mem / thresh
        else:
            outs = h2_sumspike / wins

        return outs, h2_sumspike.mean(), hebb1.data, hebb2.data, step


def mem_update_plastic(fc, alpha, beta, gamma, eta, inputs, spike, mem, hebb):
    state = fc(inputs) + alpha * inputs.mm(hebb)
    mem = (mem - thresh * spike) * mem_decay + state
    now_spike = act_fun(mem - thresh)
    hebb = w_decay * hebb - torch.bmm((inputs * beta).unsqueeze(2), ((mem / thresh) - eta).unsqueeze(1)).mean(dim=0).squeeze()

    hebb = hebb.clamp(min=-2, max=2)

    return mem, now_spike.float(), hebb


def mem_update_nonplastic(fc, inputs, spike, mem):
    state = fc(inputs)
    mem = (mem - thresh * spike) * mem_decay + state
    now_spike = act_fun(mem - thresh)

    return mem, now_spike.float()


