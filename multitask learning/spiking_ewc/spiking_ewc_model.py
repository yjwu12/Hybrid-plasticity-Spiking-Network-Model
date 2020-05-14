import torch, time, os

import torch.nn as nn
import torch.nn.functional as F
import math



# exp. configuration
exp_num = 10 # Exp. numbers
task_num = 50 # Task numbers
num_task_epochs = 10 # Epoch numbers
sparsity = 0.03 # Network sparse
cfg_fc = [1024, 1024, 10]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network parameters
si_c = 0.1
thresh = 0.5
lens = 0.5
decay = 0.8
learning_rate = 5e-4
time_window = 3
batch_size = 100

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



act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SNN_Model(nn.Module):

    def __init__(self, p=0.5, si_c=si_c):
        super(SNN_Model, self).__init__()
        self.epoch = 0.
        self.ewc_lambda = 5000  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = False  # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = None  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0  # -> keeps track of number of quadratic loss terms (for "offline EWC")

        self.fc1 = nn.Linear(28 * 28, cfg_fc[0], bias=False)
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=False)
        self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=False)
        self.mask1 = p > torch.rand(self.fc1.weight.size(), device=device)
        self.mask2 = p > torch.rand(self.fc2.weight.size(), device=device)
        self.mask1 = self.mask1.float()
        self.mask2 = self.mask2.float()


        self.alpha1 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)
        self.alpha3 = torch.nn.Parameter((torch.rand(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[1])).cuda(), requires_grad=True)
        self.eta3 = torch.nn.Parameter((.1 * torch.zeros(1, cfg_fc[2])).cuda(), requires_grad=True)

        self.gamma1 = torch.zeros(1).cuda()
        self.gamma2 = torch.zeros(1).cuda()
        self.gamma3 = torch.zeros(1).cuda()

        self.beta1 = torch.nn.Parameter((1e-3 * torch.rand(1, 784)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.beta3 = torch.nn.Parameter((1e-3 * torch.rand(1, cfg_fc[1])).cuda(), requires_grad=True)

        self.si_c = si_c  # SI: regularisation strength

    def produce_hebb(self):
        hebb1 = torch.zeros(784, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        hebb3 = torch.zeros(cfg_fc[1], cfg_fc[2], device=device)
        return hebb1, hebb2, hebb3

    def estimate_fisher(self, dataset, input_parameter=None, permutted_paramer=None, allowed_classes=None,
                        collate_fn=None):
        '''After completing training on a task, estimate diagonal of Fisher Information matrix.

        [dataset]:          <DataSet> to be used to estimate FI-matrix
        [allowed_classes]:  <list> with class-indeces of 'allowed' or 'active' classes'''

        # Prepare <dict> to store estimated Fisher Information matrix

        hebb1, hebb2, hebb3 = input_parameter

        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluation mode
        mode = self.training
        self.eval()

        # Create data-loader to give batches of size 1
        # data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda(), collate_fn=collate_fn)
        data_loader = dataset

        # Estimate the FI-matrix for [self.fisher_n] batches of size 1
        for index, (x, y) in enumerate(data_loader):
            # break from for-loop if max number of samples has been reached
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            # run forward pass of model
            # x = x.to(self._device())
            x = x.view(batch_size, -1)[:, permutted_paramer].to(device)

            outputs = self(x, hebb1=hebb1, hebb2=hebb2, hebb3=hebb3) if allowed_classes is None else self(x,
                                                                                                          hebb1=hebb1,
                                                                                                          hebb2=hebb2,
                                                                                                          hebb3=hebb3)[
                                                                                                     :, allowed_classes]
            output = outputs[0]
            if self.emp_FI:
                # -use provided label to calculate loglikelihood --> "empirical Fisher":
                label = torch.LongTensor([y]) if type(y) == int else y
                if allowed_classes is not None:
                    label = [int(np.where(i == allowed_classes)[0][0]) for i in label.numpy()]
                    label = torch.LongTensor(label)
                label = label.to(self._device())
            else:
                # -use predicted label to calculate loglikelihood:
                label = output.max(1)[1]
            # calculate negative log-likelihood
            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            # Calculate gradient of negative loglikelihood
            self.zero_grad()
            negloglikelihood.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2

        # Normalize by sample size used for estimation
        est_fisher_info = {n: p / index for n, p in est_fisher_info.items()}

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     p.detach().clone())
                # -precision (approximated by diagonal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer(
                    '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                    est_fisher_info[n])


        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1
        self.train(mode=mode)

    def ewc_loss(self):
        '''Calculate EWC-loss.'''
        if self.EWC_task_count > 0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        # If "online EWC", apply decay-term to the running sum of the Fisher Information matrices
                        fisher = self.gamma * fisher if self.online else fisher
                        # Calculate EWC-loss
                        losses.append((fisher * (p - mean) ** 2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=device)

    def update_omega(self, W, epsilon=1e-8):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change ** 2 + epsilon)
                try:
                    omega = getattr(self, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.register_buffer('{}_SI_omega'.format(n), omega_new)

    def forward(self, input, hebb1, hebb2,hebb3, wins=time_window, label=None):
        # self.mask_weight()
        batch_size = input.size(0)
        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(batch_size, cfg_fc[2], device=device)

        for step in range(wins):
            x = input
            x = x.view(batch_size, -1).float()

            h1_mem, h1_spike, hebb1 = mem_update(self.fc1,
                                                 x, h1_spike, h1_mem, hebb1)


            h2_mem, h2_spike, hebb2 = mem_update(self.fc2,
                                                 h1_spike, h2_spike, h2_mem, hebb2)


        outs = h2_mem.mm(self.fc3.weight.t())
        return outs, h1_sumspike, h1_mem, hebb1.data, hebb2.data, hebb3.data, self.eta1, self.eta2


def mem_update(fc,  inputs, spike, mem, hebb, v_th = thresh):
    total_weight = (fc.weight.t())
    state = inputs.mm(total_weight)
    mem = (mem - spike * v_th) * decay + state
    now_spike = act_fun(mem - v_th)


    return mem, now_spike.float(), hebb





