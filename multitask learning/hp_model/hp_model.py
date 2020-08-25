import torch, time, os
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn

# Exp. configuration
exp_num = 10  # Exp. numbers
task_num = 50  # Task numbers
num_task_epochs = 10  # Epoch numbers
sparsity = 0.03  # Network sparse
cfg_fc = [1024, 1024, 10]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_meta_epoch  = 35

# Network parameters
 
thresh = 0.5
lens = 0.5
mem_decay = 0.8
w_decay = 0.8
learning_rate = 5e-4
time_window = 3
batch_size = 100
t_tau = 50
probs = 0.5


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


act_fun = ActFun.apply


def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


def mask_weight(p = 0.1, task_number=10):
    mask_dict = {}
    mask_buffer_list = []
    for task in range(task_number):
        mask_buffer_list = []
        mask_buffer_list.append((p > torch.rand(784, cfg_fc[0], device=device)).float())
        mask_buffer_list.append((p > torch.rand(cfg_fc[0], cfg_fc[1], device=device)).float())
        mask_buffer_list.append((p > torch.rand(cfg_fc[1], cfg_fc[2], device=device)).float())
        mask_dict[task] = mask_buffer_list

    return mask_dict


class SNN_Model(nn.Module):

    def __init__(self):
        super(SNN_Model, self).__init__()
        self.epoch = 0.
        self.fc1 = nn.Linear(28 * 28, cfg_fc[0], bias=False)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=False)
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=False)
        self.multi_heads = [] * task_num

        self.alpha1 = torch.nn.Parameter((.1 * torch.rand(1 )).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((.1 * torch.rand(1 )).cuda(), requires_grad=True)
        self.alpha3 = torch.nn.Parameter((.1 * torch.rand(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[1])).cuda(), requires_grad=True)
        self.eta3 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[2])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((.1 * torch.zeros(1)).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((.1 * torch.zeros(1)).cuda(), requires_grad=True)
        self.gamma3 = torch.nn.Parameter((.1 * torch.zeros(1)).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-3 * torch.rand(1, 784)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.beta3 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[1])).cuda(), requires_grad=True)

    def produce_hebb(self):
        hebb1 = torch.zeros(784, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        hebb3 = torch.zeros(cfg_fc[1], cfg_fc[2], device=device)
        return hebb1, hebb2, hebb3

    def parameter_split(self):
        base_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'fv':
                base_param.append(p)

        local_param = list(set(self.parameters()) - set(base_param))
        return base_param, local_param

    def forward(self, input, hebb1, hebb2, hebb3, wins=time_window, label=None, mask_dict=None, task_index=None):

        hebb1, hebb2, hebb3 =self.produce_hebb()
        batch_size = input.size(0)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)

        for step in range(wins):
            decay_factor = np.exp(-step / t_tau)

            x = input
            x = x.view(batch_size, -1).float()


            h1_mem, h1_spike, hebb1 = mem_update(self.fc1, mask_dict[0], self.alpha1, self.beta1,
                                                 self.gamma1, self.eta1,
                                                 x * decay_factor, h1_spike, h1_mem, hebb1)

            h2_mem, h2_spike, hebb2 = mem_update(self.fc2, mask_dict[1], self.alpha2, self.beta2,
                                                 self.gamma2, self.eta2,
                                                 h1_spike * decay_factor, h2_spike, h2_mem, hebb2)

        outs = h2_mem.mm(self.fc3.weight.t() * mask_dict[2])
        return outs, h1_sumspike, h1_mem, hebb1.data, hebb2.data, hebb3.data, 0, 0.


#
def mem_update(fc, mask, alpha, beta, gamma, eta, inputs, spike, mem, hebb, v_th=thresh):
    # total_weight = (fc.weight.t() + alpha * inputs.mm(hebb) )
    state = inputs.mm(fc.weight.t() * mask + alpha * hebb * (1 - mask )) # Here we use subtration-resetting mechanism
    mem = (mem - spike * v_th) * mem_decay + state
    now_spike = act_fun(mem - v_th)
    hebb = w_decay * hebb + torch.bmm((inputs * beta).unsqueeze(2),
                                      ((mem / thresh) - eta).tanh().unsqueeze(1)).mean(dim=0).squeeze()

    return mem, now_spike.float(), hebb