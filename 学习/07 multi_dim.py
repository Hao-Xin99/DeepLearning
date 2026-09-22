import torch
import numpy as np
import matplotlib.pyplot as plt

epoch_log=[]
loss_log=[]

# 读取数据
xy=np.loadtxt('./数据集/07_diabetes/diabetes.csv',delimiter=',',dtype=np.float32)
x_data=torch.from_numpy(xy[:,:-1])
y_data=torch.from_numpy(xy[:,[-1]])
#模型构建
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        #Sigmoid 本身就能把输出归类到0~1
        return x

model=Model()

#损失函数和优化器
criterion=torch.nn.BCELoss(reduction='mean')
opitimizer=torch.optim.Adam(model.parameters(),lr=0.2)

#训练循环
for epoch in range(100):
    #前馈
    opitimizer.zero_grad()
    y_pre=model(x_data)

    #反馈
    loss=criterion(y_pre,y_data)
          #可视化
    epoch_log.append(epoch)
    loss_log.append(loss.item())
    print(epoch,loss.item())

    loss.backward()

    #更新
    opitimizer.step()


plt.plot(epoch_log,loss_log)
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()
