import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    # 在这里输入神经网络的参数
    # 这里往下的两行内容是继承torch.nn.Module的必备套路
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        # 隐藏层
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        # 预测层
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 这里是真正开始搭建神经网络，之前__inti__只是初始化这个神经网络的结构？
    # 这里的x是输入的数据
    def forward(self, x):
        # x -> hidden_layer -> activation_function
        # 激活函数ReLu其实起到了一个截断的作用，把小于的0的部分都变成了0
        x = F.relu(self.hidden(x))
        # data out of activation function -> predict_layer -> result
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)
plt.ion()  # 设置为实时打印的过程
plt.show()

# lr为learning rate
# 莫烦老师的0.5的学习率设的有点高，导致不收敛，调低到0.1就可以了，同时别忘了增加训练步数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# MSE主要是用于回归问题
loss_func = torch.nn.MSELoss()

# 训练步数
for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    # 每做完一次之后，把梯度归零一下，不然会一直保存在optimizer里面
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    optimizer.step()

    # 学习过程的可视化
    if t % 100 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
