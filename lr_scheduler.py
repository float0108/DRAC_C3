"""Learning Rate Schedulers"""
from __future__ import division
from math import pi, cos

class LRScheduler(object):
    def __init__(self, optimizer, niters, args):
        super(LRScheduler, self).__init__()

        self.mode = args.lr_mode
        self.warmup_mode = args.warmup_mode if  hasattr(args,'warmup_mode')  else 'linear'
        assert(self.mode in ['step', 'poly', 'cosine'])
        assert(self.warmup_mode in ['linear', 'constant'])

        self.optimizer = optimizer

        self.base_lr = args.lr if hasattr(args,'lr')  else 1e-3
        self.learning_rate = self.base_lr
        self.niters = niters

        self.step = [int(i) for i in args.step.split(',')] if hasattr(args,'step')  else [30, 60, 90]
        self.decay_factor = args.decay_factor if hasattr(args,'decay_factor')  else 0.1
        self.targetlr = args.targetlr if hasattr(args,'targetlr')  else 0.0
        self.power = args.power if hasattr(args,'power')  else 2.0
        self.warmup_lr = args.warmup_lr if hasattr(args,'warmup_lr')  else 0.0
        self.max_iter = args.epochs * niters
        self.warmup_iters = (args.warmup_epochs if hasattr(args,'warmup_epochs')  else 0) * niters

    def update(self, i, epoch):
        T = epoch * self.niters + i
        assert (T >= 0 and T <= self.max_iter)

        if self.warmup_iters > T:
            # Warm-up Stage
            if self.warmup_mode == 'linear':
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                    T / self.warmup_iters
            elif self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_lr
            else:
                raise NotImplementedError
        else:
            if self.mode == 'step':
                count = sum([1 for s in self.step if s <= epoch])
                self.learning_rate = self.base_lr * pow(self.decay_factor, count)
            elif self.mode == 'poly':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    pow(1 - (T - self.warmup_iters) / (self.max_iter - self.warmup_iters), self.power)
            elif self.mode == 'cosine':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    (1 + cos(pi * (T - self.warmup_iters) / (self.max_iter - self.warmup_iters))) / 2
            else:
                raise NotImplementedError

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.learning_rate

        return self.learning_rate