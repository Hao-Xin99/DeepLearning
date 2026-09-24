'''
#演示计算的进行
in_channels,out_channels=5,10
width,hight=10,10
kernel_size=3
batch_size=1

input=torch.randn(batch_size,in_channels,width,hight)
conv_layer=torch.nn.Conv2d(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    )
output=conv_layer(input)
print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)

'''

'''
#计算
input=[3,4,6,5,7,
       2,4,6,8,2,
       1,6,7,8,4,
       9,7,4,6,2,
       3,7,5,4,1]
input=torch.Tensor(input).view(1,1,5,5)

conv_layer=torch.nn.Conv2d(1,1,kernel_size=3,bias=False)

Kernel=torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
conv_layer.weight.data=Kernel.data

output=conv_layer(input)

print(output)

'''

#######MNIST用卷积实现，变动：Net变为卷积，把数据分配到GPU
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
class Net(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(kernel_size=2)
        self.linear=torch.nn.Linear(320,10)
        

    def forward(self,x):
        batch_size=x.size(0)
        x=self.pooling(F.relu(self.conv1(x)))  #卷积Conv2d也是线性层，需要激活，激活和池化顺序没要求
        x=self.pooling(F.relu(self.conv2(x)))
        #改变形状、
        x=x.view(batch_size,-1)
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
    print('Accuracy:%d %%'%(100*correct/total))


def main ():
    for epoch in range(10):
        train(epoch)

    test()


if __name__ =='__main__':
    main()



