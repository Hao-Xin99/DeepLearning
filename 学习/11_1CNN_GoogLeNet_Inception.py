#######MNIST用GoogLeNet的Inception；思想：由于不知道哪一种卷积核好用，那就都用并封装在一起（函数），好用的权重自然大
##库加载
import torch
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import torch.nn.functional as F

##Prepare Data  数据集准备
batch_size=64
path_to_MINIST='C:/Users/17740/Desktop/Pytorch/学习/数据集/09MNIST'
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
class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.branch1x1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        
        self.branch5x5_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2=torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch_pool=torch.nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
            branch1x1=self.branch1x1(x)
            
            branch5x5=self.branch5x5_1(x)
            branch5x5=self.branch5x5_2(branch5x5)
    
            branch3x3=self.branch3x3_1(x)
            branch3x3=self.branch3x3_2(branch3x3)
            branch3x3=self.branch3x3_3(branch3x3)
    
            branch_pool=torch.nn.functional.avg_pool2d(x,kernel_size=3,padding=1,stride=1)
            branch_pool=self.branch_pool(branch_pool)
    
            outputs=[branch1x1,branch5x5,branch3x3,branch_pool]
            return torch.cat(outputs,dim=1)


class Net(torch.nn.Module):
    def __init__(self,):

        super().__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(88,20,kernel_size=5)

        self.incep1=InceptionA(in_channels=10)
        self.incep2=InceptionA(in_channels=20)

        self.mp=torch.nn.MaxPool2d(2)
        self.linear=torch.nn.Linear(1408,10)
        
    def forward(self,x):
        in_size=x.shape[0]
        x=F.relu(self.mp(self.conv1(x)))
        x=self.incep1(x)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.incep2(x)
        x=x.view(in_size,-1)
        x=self.linear(x)
        return x
    


model=Net()
device=torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
#把模型放到gpu
model.to(device)


##损失函数和优化器
criterion=torch.nn.CrossEntropyLoss(reduction='mean')
optimizer=torch.optim.SGD(params=model.parameters(),lr=0.03,momentum=0.5)  #momentum是惯性，数值表示历史梯度所占比例，能累计梯度加快训练

##训练和测试
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader):
        #把训练数据送到gpu
        inputs=data[0].to(device)
        label=data[1].to(device)

        optimizer.zero_grad()
        #前馈
        y_pred=model(inputs)
        loss=criterion(y_pred,label)
        #反馈
        loss.backward()
        #更新
        optimizer.step()

        running_loss+=loss.item()
        if batch_idx %300 ==299:
            #每300轮更新，算平均损失
            print('[%d,%5d] loss:%.3f'% (epoch+1,batch_idx+1,running_loss/300))
            running_loss=0

def test():
    correct=0
    total=0
    with torch.no_grad():
        for (inputs,labels) in test_loader:
            #把数据送到gpu
            inputs,labels=inputs.to(device),labels.to(device)

            outputs=model(inputs)
            #outputs是（batch_size,10）两个维度，max函数输出的是两个张量构成的元组(value,indices), value:最大值的数，indices:最大值的列数 
            #indices形状(batch_size,),batch_size行1列，与labels一致
            #dim=0是行方向，dim=1是列，dim=1,把列压缩成一列
            _,pred=torch.max(outputs,dim=1)  #pred是一个张量
            total+=labels.size(0)
            correct+= (pred==labels).sum().item()
    print('Accuracy:%d %% [%d/%d]'%((100*correct/total),correct,total))


def main ():
    for epoch in range(10):
        train(epoch)

    test()


if __name__ =='__main__':
    main()









