import torch

x = torch.Tensor([-4.0]).double()
x.requires_grad = True
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()
xpt, ypt = x, y

# forward pass went well
print(ypt.data.item())
# backward pass went well
print(xpt.grad.item())



