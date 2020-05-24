import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# Hyper parameter
EPOCH = 2
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False  # 如果还没下载，这里写成是True

# 下载MNIST训练集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# plot one example
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title("%i" % train_data.train_labels[0])
# plt.show()

# 如果开启多线程的话，可以加一个 num_workers=2
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 这里train设置成False说明提取出来的是test data
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 为了节约时间, 我们测试时只测试前2000个
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# unsqueeze是为数据增加一个维度
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(              # input (1, 28, 28)
                in_channels=1,      # 图片是灰度图，只有一个通道，或者说图像的厚度为1
                out_channels=16,    # out_channels的个数就是Filter的个数，这也就是为什么一层卷积过后，变厚了很多
                kernel_size=5,      # Filter的大小为5x5
                stride=1,           # Filter在图像上移动的时候，每次上下平移的pixel的个数
                padding=2,          # 给图片加一个外框，数值为0。padding = (kernal_size - 1)/2 = 2
            ),                              # 卷积层 -> (16, 28, 28)
            nn.ReLU(),                      # 激活函数 -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),    # 池化层，划分成在2x2的格子，然后取max -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input (16, 14, 14)
            # 这里就不具体写 xxx = xxx 了
            nn.Conv2d(16, 32, 5, 1, 2),     # -> (32, 14, 14)
            nn.ReLU(),                      # -> (32, 14, 14)
            nn.MaxPool2d(2)                 # -> (32, 7, 7)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # -> (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)   # 这里 x.size(0) 是什么？
        output = self.out(x)        # -> (batch, 32 * 7 * 7)
        return output, x


cnn = CNN()
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


# 下面是有关可视化的代码
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)


plt.ion()


# 训练
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]            # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # back propagation, compute gradients
        optimizer.step()                # apply gradients

        if step % BATCH_SIZE == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)

plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')