import numpy as np
import torch

'''
#这一段是Softmax的实现机制
y=np.array([1,0,0])
z=np.array([0.2,0.1,-0.1])
y_pre=np.exp(z)/np.exp(z).sum()
loss=(-y*np.log(y_pre)).sum()
print(loss)
'''

y=torch.LongTensor([0])
z=torch.Tensor([[0.2,0.1,-0.1]])
criterion=torch.nn.CrossEntropyLoss()
loss=criterion(z,y)
print(loss)




