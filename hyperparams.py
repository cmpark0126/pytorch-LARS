# -*- coding: utf-8 -*-

class Base:
    batch_size = 128 # initial batch size
    lr = 0.05 # Initial learning rate
    multiples = 1 # help to calculate k

class Hyperparams:
    '''Hyper parameters'''
    device = [0]
    
    batch_size = Base.batch_size * (2 ** (Base.multiples - 1)) # k = (2 ** (Base.multiples - 1)
    lr = Base.lr * (2 ** (Base.multiples - 1)) # for LR linear scailing

    # optim
    momentum = 0.9
    weight_decay = 5e-4   
    trust_coef = 0.1
    
    # warm-up step & Linear Scaling Rule
    warmup_multiplier = 2
    warmup_epoch = 5
    
    # decay lr step (polynomial)
    max_decay_epoch = 200
    end_learning_rate = 0.0001
    
    num_of_epoch = 200
    with_lars = False
    resume = False
    
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
    
class Hyperparams_for_val:
    checkpoint_folder_name = 'checkpoint'
    with_lars = False
    batch_size = 128
    device = [0]