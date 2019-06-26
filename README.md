# Implement LARS

### Object
* link: ["Large Batch Training of Convolutional Networks (LARS)"](https://arxiv.org/abs/1708.03888)
* 위 논문에 소개된 LARS를 PyTorch로 구현
* Data: CIFAR10

### Usage
```bash
$ git clone https://github.com/cmpark0126/pytorch-LARS.git
$ cd pytorch-LARS/
$ vi hyperparams.py # hyperparameter 및 config용 요소 확인. 필요시 수정 가능
$ python train.py # CIFAR10 학습 시작
$ python val.py # 학습 결과 확인
```

### Result
##### Attempt 1
##### Attempt 2
##### Attempt 3
##### Attempt 4
##### Attempt 5

### Reference
* Base code: https://github.com/kuangliu/pytorch-cifar
* warm-up LR scheduler: https://github.com/ildoonet/pytorch-gradual-warmup-lr/tree/master/warmup_scheduler
    * 이를 기반으로 PolynomialLRDecay class 구현
    * polynomial LR decay scheduler
    * 참고: scheduler.py
* Pytorch Doc / Optimizer: https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html
    * Optimizer class
    * SGD class
