import numpy as np
import torch
import torch.optim as optim


def where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)


def cosine_decay_with_warmup(global_step: int,
                             learning_rate_base: float,
                             total_steps: int,
                             warmup_learning_rate: float = 0.0,
                             warmup_steps: int = 0,
                             hold_base_rate_steps: int = 0):
    """https://arxiv.org/abs/1608.03983
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / (total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps < 0:
        learning_rate = where(global_step > warmup_steps + hold_base_rate_steps, learning_rate,
                              learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = where(global_step < warmup_steps, warmup_rate, learning_rate)
    result = where(global_step > total_steps, 0.0, learning_rate)
    return result


class SuperConvergence(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, base_lr,
                 warmup_learning_rate, warmup_steps, hold_base_rate_steps):
        self._total_steps = total_steps
        self._max_lr = base_lr
        self._warmup_lr = warmup_learning_rate
        self._warmup_steps = warmup_steps
        self._hold_base_rate_steps = hold_base_rate_steps
        super().__init__(optimizer)

        self._lr_base = self.base_lrs

    def get_lr(self):
        return [cosine_decay_with_warmup(self.last_epoch, self._max_lr, self._total_steps,
                                         self._warmup_lr, self._warmup_steps,
                                         self._hold_base_rate_steps)
                for base_lr in self.base_lrs]
