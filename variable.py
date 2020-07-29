import torch
from torch.autograd import Variable

# tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
tensor = torch.zeros(size=[3, 3])
print(tensor)
print(tensor.size()[1])

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

# indices：非零元素的坐标，如下例的indices，indices[0]表示三个横坐标，indices[1]表示三个纵坐标，因此稀疏矩阵中非零的位置分别为（0，2），（1，0），（1，2）
# values: 非零元素的值， 如下例中的values，三个非零元素分别为3，4，5
# size: 次数矩阵的维度

indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
values = torch.tensor([3, 4, 5], dtype=torch.float32)
x = torch.sparse_coo_tensor(indices, values, [2, 4])
print(indices)
print(values)
print(x)

step_arr = torch.linspace(0, 9, steps=10)
print(step_arr)