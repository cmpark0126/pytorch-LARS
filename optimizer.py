import torch
from torch.optim.optimizer import Optimizer, required

class SGD_without_lars(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SGD_without_lars, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_without_lars, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                #torch.cuda.nvtx.range_push('trial')
                if p.grad is None:
                    continue
                d_p = p.grad.data
                torch.cuda.nvtx.range_push('weight decay')
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                torch.cuda.nvtx.range_pop()
                # d_p.mul_(lr)

                torch.cuda.nvtx.range_push('momentum')
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push('weight update')
                p.data.add_(-lr, d_p)
                torch.cuda.nvtx.range_pop()
                
                # torch.cuda.nvtx.range_pop()
        return loss


class SGD_with_lars(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, trust_coef=1.): # need to add trust coef
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if trust_coef < 0.0:
            raise ValueError("Invalid trust_coef value: {}".format(trust_coef))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coef=trust_coef)

        super(SGD_with_lars, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_with_lars, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coef = group['trust_coef']
            global_lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p_norm = torch.norm(p.data, p=2)
                d_p_norm = torch.norm(d_p, p=2).add_(momentum, p_norm)
                lr = torch.div(p_norm, d_p_norm).mul_(trust_coef)

                lr.mul_(global_lr)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                d_p.mul_(lr)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(-1, d_p)

        return loss


class SGD_with_lars_ver2(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, trust_coef=1.): # need to add trust coef
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if trust_coef < 0.0:
            raise ValueError("Invalid trust_coef value: {}".format(trust_coef))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coef=trust_coef)

        super(SGD_with_lars_ver2, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_with_lars_ver2, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            trust_coef = group['trust_coef']
            global_lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # torch.cuda.nvtx.range_push('p_norm')
                p_norm = torch.norm(p.data, p=2)
                # torch.cuda.nvtx.range_pop()
#                 print('p_norm')
#                 print(p_norm)
                # torch.cuda.nvtx.range_push('d_p_norm')
                d_p_norm = torch.norm(d_p, p=2).add_(weight_decay, p_norm)
                #torch.cuda.nvtx.range_pop()
#                 print('d_p_norm')
#                 print(torch.norm(d_p, p=2))
                #torch.cuda.nvtx.range_push('div')
                lr = torch.div(p_norm, d_p_norm)
                #torch.cuda.nvtx.range_pop()
#                 print('result')
#                 print(torch.div(p_norm, d_p_norm))
#                 print('')

                
                #torch.cuda.nvtx.range_push('calculate local lr')
                lr.mul_(-global_lr*trust_coef)
                #torch.cuda.nvtx.range_pop()

                #torch.cuda.nvtx.range_push('weight decay')
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                #torch.cuda.nvtx.range_pop()

                #torch.cuda.nvtx.range_push('momentum')
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                #torch.cuda.nvtx.range_pop()

                #torch.cuda.nvtx.range_push('weight update')
                d_p.mul_(lr)
                p.data.add_(d_p)
                #torch.cuda.nvtx.range_pop()
                

        return loss
