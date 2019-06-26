# -*- coding: utf-8 -*-

class Base:
    batch_size = 128 # initial batch size
    lr = 0.15 # Initial learning rate
    
    multiples = 7

class Hyperparams:
    '''Hyper parameters'''
    device = [Base.multiples - 1]
    
    batch_size = Base.batch_size * (2 ** (Base.multiples - 1))
    lr = Base.lr 

    # optim
    momentum = 0.9
    trust_coef = 0.1
    
    # warm-up step & Linear Scaling Rule
    warmup_multiplier = (2 ** (Base.multiples - 1)) 
    warmup_epoch = 5
    
    # decay lr step
    max_decay_epoch = 95
    end_learning_rate = 0.0001 * Base.lr * (2 ** (Base.multiples - 1))
    
    num_of_epoch = 250
    
    resume = False
    
    with_lars = False
    weight_decay = 5e-4
#     with_lars = True
#     weight_decay = 5e-3    
    
    def print_hyperparms():
        print('batch_size: ' + str(Hyperparams.batch_size))
        print('lr: ' + str(Hyperparams.lr))
        print('momentum: ' + str(Hyperparams.momentum))
        print('trust_coef: ' + str(Hyperparams.trust_coef))
        print('warmup_multiplier: ' + str(Hyperparams.warmup_multiplier))
        print('warmup_epoch: ' + str(Hyperparams.warmup_epoch))
        print('max_decay_epoch: ' + str(Hyperparams.max_decay_epoch))
        print('end_learning_rate: ' + str(Hyperparams.end_learning_rate))
        print('num_of_epoch: ' + str(Hyperparams.num_of_epoch))
        print('device: ' + str(Hyperparams.device))
        print('resume: ' + str(Hyperparams.resume))
        print('with_lars: ' + str(Hyperparams.with_lars))
        print('weight_decay: ' + str(Hyperparams.weight_decay))
    
    def get_info_dict():
        return dict(batch_size=Hyperparams.batch_size, 
                     lr=Hyperparams.lr, 
                     momentum=Hyperparams.momentum, 
                     trust_coef=Hyperparams.trust_coef, 
                     warmup_multiplier=Hyperparams.warmup_multiplier, 
                     warmup_epoch=Hyperparams.warmup_epoch, 
                     max_decay_epoch=Hyperparams.max_decay_epoch, 
                     end_learning_rate=Hyperparams.end_learning_rate, 
                     num_of_epoch=Hyperparams.num_of_epoch, 
                     device=Hyperparams.device, 
                     resume=Hyperparams.resume, 
                     with_lars=Hyperparams.with_lars, 
                     weight_decay=Hyperparams.weight_decay)