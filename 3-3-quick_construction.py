import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


# method-1
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


net1 = Net(2, 10, 2)

# ==================== 3-3-quick_construction.py 的重点 ==================== #
# Method-2
# 等效为上面的 Method-1，它给出了一种不需要自己来打造一个class的方式，使得可以快速构建一个神经网络
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
# ========================================================================== #

plt.ion()  # 设置为实时打印的过程
plt.show()

# lr为learning rate
# 莫烦老师的0.5的学习率设的有点高，导致不收敛，调低到0.1就可以了，同时别忘了增加训练步数
optimizer = torch.optim.SGD(net2.parameters(), lr=0.005)
# CrossEntropyLoss用于多分类任务
loss_func = torch.nn.CrossEntropyLoss()

# 训练步数
for t in range(500):
    # 这里“莫烦Python”的版本是 out = net(x)
    # 个人觉得比较奇怪，就改成了如下的形式：
    out = net2.forward(x)
    loss = loss_func(out, y)
    # 每做完一次之后，把梯度归零一下，不然会一直保存在optimizer里面
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    optimizer.step()

    # 学习过程的可视化
    if t % 50 == 0:
        # plot and show learning process
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        out = torch.max(F.softmax(out), 1)[1]
        pred_y = out.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
