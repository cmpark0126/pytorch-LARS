import torch
from torch.optim.optimizer import Optimizer, required

class SGD_without_lars(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
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
                if p.grad is None:
                    continue
                d_p = p.grad.data
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
                
                p_norm = torch.norm(p.data, p=2)
#                 print('p_norm')
#                 print(p_norm)
                d_p_norm = torch.norm(d_p, p=2).add_(momentum, p_norm)
#                 print('d_p_norm')
#                 print(torch.norm(d_p, p=2))
                lr = torch.div(p_norm, d_p_norm).mul_(trust_coef)
#                 print('result')
#                 print(torch.div(p_norm, d_p_norm))
#                 print('')
                
                lr.mul_(global_lr)
                
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(-lr, d_p)
    
#         input()
        return loss