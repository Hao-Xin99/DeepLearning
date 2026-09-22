####预学习
'''
#这一段是Softmax的实现机制
y=np.array([1,0,0])
z=np.array([0.2,0.1,-0.1])
y_pre=np.exp(z)/np.exp(z).sum()
loss=(-y*np.log(y_pre)).sum()
print(loss)
'''
'''
y=torch.LongTensor([0])  #y有两种表示方式,一种是one-hot编码,只写'1'所在下标,如[1],真实:[0,1,0,0],表示第二个是对的(省内存); 一种是完整的类别标签，如[0,1,2],表示
z=torch.Tensor([[0.2,0.1,-0.1]])
criterion=torch.nn.CrossEntropyLoss()
loss=criterion(z,y)
print(loss)
'''

##库加载
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.nn.functional as F

##Prepare Data  数据集准备
batch_size=64
path_to_MINIST='C:\Users\17740\Desktop\Pytorch\学习\数据集\09MNIST'
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
    ])
#数据集MIINIST
train_dataset=datasets.MNIST(root=path_to_MINIST,
                             train=True,
                             transform=transform,
                             download=False)
test_dataset=datasets.MNIST(root=path_to_MINIST,
                            train=False,
                            transform=transform,
                            download=False)
#数据加载器
train_loader=DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    )
test_loader=DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    )


##模型构建
class Net(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.Linear1=torch.nn.Linear(784,512)
        self.Linear2=torch.nn.Linear(512,256)
        self.Linear3=torch.nn.Linear(256,128)
        self.Linear4=torch.nn.Linear(128,64)
        self.Linear5=torch.nn.Linear(64,10)
        self.Relu=torch.nn.ReLU()

    def forward(self,x):
        x=x.view(-1,784)
        x=self.Relu(self.Linear1(x))
        x=self.Relu(self.Linear2(x))
        x=self.Relu(self.Linear3(x))
        x=self.Relu(self.Linear4(x))
        return self.Linear5(x)  #返回的是未经过激活的x

net=Net()

##损失函数和优化器
criterion=torch.nn.CrossEntropyLoss(reduction='mean')
optimizer=torch.optim.SGD(params=net.parameters(),lr=0.03,momentum=0.5)  #momentum是惯性，数值表示历史梯度所占比例，能累计梯度加快训练

##训练和测试
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader):
        inputs=data[0]
        label=data[1]
        optimizer.zero_grad()
        #前馈
        y_pred=net(inputs)
        loss=criterion(y_pred,label)
        #反馈
        loss.backward()
        #更新
        optimizer.step()

        if batch_idx %300 ==299:
            #每300轮更新，算平均损失
            print('[%d,%5d] loss:%3f'% (epoch+1,batch_idx+1,running_loss/300))
            running_loss=0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for (inputs,labels) in train_loader:
            outputs=net(inputs)
            pred=torch.max(outputs,dim=1)
            total+=labels.size(0)
            correct+= (pred==labels).sum().item()
    print('Accuracy:%d %%'%(100*correct/total))


def main ():
    for epoch in range(10):
        train(epoch)

    test()


if __name__ =='__main__':
    main()
