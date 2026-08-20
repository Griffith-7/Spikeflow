"""Spike-aware optimizer with binary weight constraints."""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class SpikeAdamW(Optimizer):
    """AdamW with binary regularization: pushes weights toward {-1, +1}."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        binary_reg: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, binary_reg=binary_reg)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            binary_reg = group["binary_reg"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SpikeAdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                update = (exp_avg / bias_correction1) / denom

                if wd > 0:
                    p.mul_(1 - lr * wd)

                if binary_reg > 0:
                    sign_reg = binary_reg * (p.abs() - 1.0).clamp(min=0) * p.sign()
                    update = update + sign_reg

                p.add_(update, alpha=-lr)

        return loss


class BinaryConnect(Optimizer):
    """BinaryConnect: clamp weights to [-1, 1] after each SGD step."""

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 1e-4):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                p.add_(buf, alpha=-lr)
                p.data.clamp_(-1, 1)

        return loss
