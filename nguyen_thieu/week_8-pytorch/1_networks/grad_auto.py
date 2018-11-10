# Pytorch uses Autograd library for backpropagation during training
## It also auto deveriative for us

## Tensorflow : autodiff
## Pytorch : autograd

## Run backpropagation by method: .backward()    (Only required during training)

import torch

a1 = torch.rand(2, 3)
print(a1.requires_grad)

a1.requires_grad_()         # Enable tracking for computations on this tensor
print(a1.requires_grad)

print(a1.grad)      # Accumulates the gradient of the computations w.r.t this tensor after the backward pass
print(a1.grad_fn)

print("======================")

out = a1 * a1
print(out.requires_grad)
print(out.grad)
print(out.grad_fn)

print("++++++++++++++++++++++")

out2 = (a1 * a1).mean()
print(out2.grad)
print(out2.grad_fn)

out2.backward()
print(a1.grad)
print(a1.grad_fn)

print("========================")

## Stop using grad for tensor at some moment (block of code)

with torch.no_grad():
    out3 = a1 * a1
    print(out3)
    print("The original still need grad: ", a1.requires_grad)
    print("But the new one don't need grad: ", out3.requires_grad)





