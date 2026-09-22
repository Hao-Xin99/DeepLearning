import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# device：有 GPU 就用 GPU，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 残差块（ResidualBlock）
# 这就是我们之前讲的 y = F(x) + x 的实现
# 卷积版的残差块：两层 Conv2d + BatchNorm + ReLU，最后把输入 x 直接加回来
# ============================================================
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 输入和输出通道数相同，这样才能做 out + x 加法
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x                          # ① 把输入存下来（那条捷径）
        out = self.relu(self.bn1(self.conv1(x)))   # ② 第一层卷积+BN+ReLU
        out = self.bn2(self.conv2(out))             # ③ 第二层卷积+BN（注意最后这层不加ReLU）
        out = out + residual                  # ④ 关键！把捷径的输入加回来
        out = self.relu(out)                 # ⑤ 加完再过一次 ReLU
        return out


# ============================================================
# 主网络：用残差块搭
# 输入：(batch, 1, 28, 28) 的灰度图
# 输出：(batch, 10) 的 log 概率
# ============================================================
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层：把单通道灰度图变成 16 通道的特征图
        # Conv2d(输入通道, 输出通道, 卷积核大小, padding=1 保持尺寸不变)
        self.conv_in = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn_in = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()

        # 两个残差块，通道数都是 16
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)

        # 全局平均池化：把 (batch, 16, 28, 28) 压成 (batch, 16, 1, 1)
        # 意思是每个通道的 28×28 个数字取平均，变成一个数
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

        # 最后全连接：16 个特征 -> 10 个数字类别
        self.fc = torch.nn.Linear(16, 10)

    def forward(self, x):
        # 注意：这里 x 进来就是 (batch, 1, 28, 28)，不需要 view 拉平！
        # 卷积网络直接吃二维图片，保留空间结构
        x = self.relu(self.bn_in(self.conv_in(x)))   # (batch, 16, 28, 28)
        x = self.block1(x)                            # (batch, 16, 28, 28)
        x = self.block2(x)                             # (batch, 16, 28, 28)
        x = self.pool(x)                               # (batch, 16, 1, 1)
        x = x.view(x.size(0), -1)                      # (batch, 16)
        x = torch.nn.functional.log_softmax(self.fc(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


# 测试评估
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    net.eval()   # 评估模式：BatchNorm 用全局统计，dropout 关闭
    with torch.no_grad():
        for (x, y) in test_data:
            x = x.to(device)
            y = y.to(device)
            # 卷积网络直接吃 (batch, 1, 28, 28)，不用 view 拉平
            outputs = net(x)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    net.train()  # 切回训练模式
    return n_correct / n_total


def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    net.to(device)

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(3):
        for (x, y) in train_data:
            net.zero_grad()
            x = x.to(device)
            y = y.to(device)
            # 卷积网络直接吃 (batch, 1, 28, 28)
            output = net(x)
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 预测 5 张测试集图片，并排显示，标题写预测和真实标签
    net.eval()
    with torch.no_grad():
        for i in range(5):
            image, label = test_data.dataset[i]          # 取第 i 张
            image_gpu = image.to(device)
            predict = torch.argmax(net(image_gpu.unsqueeze(0)))  # 预测
            plt.subplot(1, 5, i + 1)                      # 1行5列，画第 i+1 个
            plt.imshow(image.view(28, 28))
            # 标题：预测对了就绿色，错了就红色
            color = "green" if int(predict) == int(label) else "red"
            plt.title(f"pred: {int(predict)}\ntrue: {int(label)}", color=color)
            plt.axis("off")                               # 不画坐标轴
    plt.show()
    net.train()


if __name__ == "__main__":
    main()
