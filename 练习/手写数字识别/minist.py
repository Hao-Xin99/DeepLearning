import torch
#1.utils是一个工具模块，提供了数据加载、模型保存等功能,utils.data是一个子模块，提供了数据加载相关的功能
#2.Dataloader是一个数据加载器，用于批量加载数据
#3. torchvision是一个计算机视觉库，提供了常用的数据集（dataset）、模型和图像处理工具
#4.transforms是一个图像处理模块，提供了常用的图像变换操作
#5.datasets是一个数据集模块，提供了常用的数据集类，如MNIST、CIFAR-10等,是用torch.utils.data里的Dataset基弄好的成品
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

#nn是一个神经网络模块，提供了常用的神经网络层和损失函数
#Module是一个基类，所有的神经网络模型都应该继承自这个类
#Linear是一个全连接层，接受输入特征数和输出特征数作为参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    #x的特征（最后一维度）必须是28*28=784
    def forward(self, x):  
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        #log_softmax是一个对数softmax函数，用于将输出转换为概率分布，再取对数，dim=1表示在第1维（矩阵的行）上进行softmax操作
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)


#测试评估
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x = x.to(device)
            y = y.to(device)
            outputs = net(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
#train_data与test_data都是DataLoader对象，用于处理数据（分批，打乱等），里面有dataset对象，dataset对象是MNIST数据集的实例，里面有数据和标签
#所以能通过for循环遍历train_data和test_data，得到每一批数据的输入x和标签y
#x是输入的图像，通过transforms.ToTensor()转换为张量，形状为(batch_size(15), 1, 28, 28)
#y是标签，形状为(batch_size(15),)
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    
    net.to(device)


    print("initial accuracy:", evaluate(test_data, net))
    #优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(3):
        for (x, y) in train_data:
            net.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))


    #预测一张：
    image,label=test_data.dataset[2]
    #view函数是pytorch自带的方法
    image_gpu=image.to(device)
    predict=torch.argmax(net(image_gpu.view(-1,28*28)))
    plt.figure(0)
    plt.imshow(image.view(28,28))
    plt.title("prediction: "+str(int(predict)))
    plt.show()


'''
预测一堆的数据
for (n, (x, _)) in enumerate(test_data):
        if n > 4:
            break
        predict = torch.argmax(net(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()
'''
    


if __name__ == "__main__":
    main()