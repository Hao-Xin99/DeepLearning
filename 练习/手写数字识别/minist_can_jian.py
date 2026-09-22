import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 残差块（用你已经懂的 Linear 实现）
# 输入输出维度相同（都是 64），这样才能做 out + x 加法
# ============================================================
class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, dim)   # 全连接层
        self.fc2 = torch.nn.Linear(dim, dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x                       # ① 把输入存下来（那条捷径）
        out = self.relu(self.fc1(x))      # ② 走第一层
        out = self.fc2(out)               # ③ 走第二层
        out = out + residual              # ④ 关键！把捷径的输入加回来
        out = self.relu(out)              # ⑤ 加完再过一次 ReLU
        return out


# ============================================================
# 主网络：和原来 minist.py 几乎一样，只把中间两层换成残差块
# 原来：fc1(784→64) → fc2(64→64) → fc3(64→64) → fc4(64→10)
# 现在：fc_in(784→64) → block1(残差块) → block2(残差块) → fc_out(64→10)
# ============================================================
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = torch.nn.Linear(28*28, 64)     # 输入层：784 → 64
        self.block1 = ResidualBlock(64)             # 残差块1（代替原来的 fc2）
        self.block2 = ResidualBlock(64)             # 残差块2（代替原来的 fc3）
        self.fc_out = torch.nn.Linear(64, 10)       # 输出层：64 → 10

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc_in(x))   # 输入层 + ReLU
        x = self.block1(x)                            # 残差块1
        x = self.block2(x)                            # 残差块2
        x = torch.nn.functional.log_softmax(self.fc_out(x), dim=1)  # 输出层
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    net.eval()
    with torch.no_grad():
        for (x, y) in test_data:
            x = x.to(device)
            y = y.to(device)
            outputs = net(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    net.train()
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
            output = net(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    # 预测 5 张
    net.eval()
    with torch.no_grad():
        plt.figure(figsize=(15, 3))
        for i in range(5):
            image, label = test_data.dataset[i]
            image_gpu = image.to(device)
            predict = torch.argmax(net(image_gpu.view(-1, 28*28)))
            plt.subplot(1, 5, i + 1)
            plt.imshow(image.view(28, 28))
            color = "green" if int(predict) == int(label) else "red"
            plt.title(f"pred:{int(predict)}\ntrue:{int(label)}", color=color)
            plt.axis("off")
        plt.tight_layout()
        plt.show()
    net.train()


if __name__ == "__main__":
    main()
