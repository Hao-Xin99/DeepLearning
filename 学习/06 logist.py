import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# 准备数据集
x_data=torch.Tensor([[1.0], [2.0], [3.0]])
y_data=torch.Tensor([[0],[0],[1]])

# 定义模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred

model=LogisticRegressionModel()

# 定义损失函数和优化器
criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    optimizer.zero_grad()
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    loss.backward()
    optimizer.step()


# model训练后参数更新，用新模型预测，并h画出训练结果
the_x=np.linspace(0,10,200)
x_t=torch.Tensor(the_x).view(-1,1)
y_t=model(x_t)
the_y=y_t.data.numpy()
plt.plot(the_x,the_y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Pass Probability')
plt.grid()
plt.show()



