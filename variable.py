import torch
from torch.autograd import Variable

tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# use variable to construct backward propagation
variable = Variable(tensor, requires_grad=True)

# print(tensor)
# print(variable)

t_out = torch.mean(tensor*tensor)  # x^2
v_out = torch.mean(variable*variable)

# print(t_out)
# print(v_out)

v_out.backward()

print(variable.grad)
print(variable.data)
print(variable.data.numpy())
