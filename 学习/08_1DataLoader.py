import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

#加载数据集
class DiabetesDataset(Dataset):
    def __init__(self,datapath):
        xy=np.loadtxt(datapath,delimiter=',',dtype=np.float32)
        ####不要忘了装载的是numpy类型，要转回去。
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])
        self.len=xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

My_Dataset=DiabetesDataset('./数据集/07_diabetes/diabetes.csv')  #实例化

#数据装载器实例化
#num_workers是让电脑开启多进程来处理数据，进程间互不干扰。但由于Windows无Fork,需要写那么那个“if name == "__main__":”
My_DataLoador=DataLoader(My_Dataset,batch_size=8,shuffle=True,num_workers=2)

##构建模型
class Model(torch.nn.Module):
    def __init__(self,):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x

model=Model()     #实例化

#损失函数和优化器
criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#训练循环
if __name__ == "__main__":
    for epoch in range(100):
        for i,data in enumerate(My_DataLoador):
            #把输入输出分开
            input=data[0]
            label=data[1]

            #前向
            y_Pre=model(input)
            loss=criterion(y_Pre,label)
            print(epoch,i,loss.item())
            #反向
            optimizer.zero_grad()
            loss.backward()

            #更新：
            optimizer.step()

















