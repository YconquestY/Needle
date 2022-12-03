"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            #if p.grad is None:
            #    continue
            p_id = id(p)
            grad = (1. - self.momentum) * (p.grad.data + self.weight_decay * p.data) + \
                   self.momentum * self.u.get(p_id, 0.)
            p.data -= self.lr * grad
            self.u[p_id] = grad
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            #if p.grad is None:
            #    continue
            p_id = id(p)
            grad = p.grad.data + self.weight_decay * p.data
            self.m[p_id] = self.beta1 * self.m.get(p_id, 0.) + (1. - self.beta1) * grad
            self.v[p_id] = self.beta2 * self.v.get(p_id, 0.) + (1. - self.beta2) * grad ** 2.
            m_hat = self.m[p_id] / (1. - self.beta1 ** self.t)
            v_hat = self.v[p_id] / (1. - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat ** .5 + self.eps)
        ### END YOUR SOLUTION
