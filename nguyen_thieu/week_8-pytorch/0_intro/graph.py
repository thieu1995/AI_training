# Similar to TF graphs of TensorFlow but computation graphs in Pytorch are dynamic

# https://app.pluralsight.com/player?course=pytorch-building-deep-learning-models&author=janani-ravi&name=f20b94f1-48ff-4232-bbbf-09a92f75ade9&clip=8&mode=live
import torch
from torch.autograd import Variable         # Same Tensor

x = Variable(torch.randn(1, 10))
h = Variable(torch.randn(1, 20))
W_h = Variable(torch.randn(20, 20))
W_x = Variable(torch.randn(20, 10))

h_prod = torch.mm(W_h, h.t())       # h production
x_prod = torch.mm(W_x, x.t())       # x production

next_h = (h_prod + x_prod).tanh()
loss = next_h.sum()
loss.backward()                     ## Gradient descents
