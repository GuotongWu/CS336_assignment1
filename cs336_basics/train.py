from typing import Callable, Iterable, Optional
import torch
import math

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    batch = inputs.shape[0]
    inputs = inputs - torch.max(inputs, dim=1, keepdim=True).values
    numerate = inputs[torch.arange(batch), targets]
    denominator = torch.log(torch.sum(torch.exp(inputs), dim=1))
    return torch.mean(denominator - numerate)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter] | list[dict], lr: float=1e-5, 
                 weight_decay: float=0.01, betas: tuple=(0.99, 0.999), eps: float=1e-5):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.state["t"] = 1
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = dict(exp_avg=torch.zeros(p.shape), exp_avg_sq=torch.zeros(p.shape))  

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            t = self.state["t"]
            new_lr = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))

            for p in group["params"]:
                if p.grad is None:
                    continue 

                state = self.state[p]
                self.state[p]["exp_avg"] = exp_avg = beta1 * state["exp_avg"] + (1 - beta1) * p.grad
                self.state[p]["exp_avg_sq"] = exp_avg_sq = beta2 * state["exp_avg_sq"] + (1 - beta2) * (p.grad ** 2)
                
                p.data -= new_lr * exp_avg / torch.sqrt(exp_avg_sq + group["eps"])
                p.data -=  lr * group["weight_decay"] * p.data
            
        self.state["t"] += 1
        return loss

def lr_cosine_sheduling(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (1 + 
            math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float=1e-6):
    l2_norm = 0
    for p in parameters:
        if p.grad is not None:
            l2_norm += p.grad.norm(2).pow(2)

    l2_norm = l2_norm ** 0.5
    
    if l2_norm > max_l2_norm:
        coff = max_l2_norm / (l2_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(coff)

if __name__ == "__main__":
    # inputs = torch.rand(32, 50)
    # targets = torch.randint(0, 50, (32,))

    # weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = SGD([weights], lr=1e3)
    # for t in range(100):
    #     opt.zero_grad()
    #     # Reset the gradients for all learnable parameters.
    #     loss = (weights**2).mean() # Compute a scalar loss value.
    #     print(loss.cpu().item())
    #     loss.backward() # Run backward pass, which computes gradients.
    #     opt.step() # Run optimizer step.
    

    # weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = AdamW([weights])
    # for t in range(100):
    #     opt.zero_grad()
    #     # Reset the gradients for all learnable parameters.
    #     loss = (weights**2).mean() # Compute a scalar loss value.
    #     print(loss.cpu().item())
    #     loss.backward() # Run backward pass, which computes gradients.
    #     opt.step() # Run optimizer step.

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    loss = (weights**2).mean() 
    loss.backward()
    gradient_clipping([weights], 1)