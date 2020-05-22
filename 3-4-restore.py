import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

# 假数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    # 建网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()

    # 训练
    for t in range(500):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 保存整个图
    torch.save(net1, "3-4-net.pkl")
    # 只保存图中节点的参数
    torch.save(net1.state_dict(), "3-4-net_params.pkl")

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("Net1")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)


def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('3-4-net.pkl')
    prediction = net2(x)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title("Net2")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)


# 这种方法比restore_net更快一点
def restore_params():
    # 需要先建一个与之前相同的网络
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    net3.load_state_dict(torch.load("3-4-net_params.pkl"))
    prediction = net3(x)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title("Net3")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)


# 保存 net1 (1. 整个网络, 2. 只有参数)
save()

# 提取整个网络
restore_net()

# 提取网络参数, 复制到新网络
restore_params()
plt.show()
