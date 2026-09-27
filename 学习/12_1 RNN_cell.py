##RNN主要用来处理带有序列的，或时间的。用一个线性层，权重共享
# RNN Cell本质上是共享线性层,通过循环把历史输入信息融合起来
#(最重要的就是把维度弄清楚)


####利用RNN Cell

'''

import torch

batch_size=1
seq_len=3
input_size=4
hidden_size=2

cell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

dataset=torch.randn(seq_len,batch_size,input_size)
hidden=torch.zeros(batch_size,hidden_size)

####由于RNN可并行处理批次（多个样本），所以输入input的维度（batch_size,input_size），输出output的维度（batch_size,hidden_size）
####整个输入序列形状（seqLen,batch_size,input_size），seqLen即时间步的个数，这样规定形状，便于enumerate(dataset)取出来的数据自发的便是input的（batch_size,input_size）
for idx,input in enumerate(dataset):
    print('='*2,idx,'='*20)
    print('input size:',input.shape)

    hidden=cell(input,hidden)

    print('output size:',hidden.shape)
    print(hidden)
'''


#字母序列重排
import torch

input_size=4  #输入特征数目，每一个时间步是（1，4）的，所以特征为4
hidden_size=4
batch_size=1

idx2char=['e','h','l','o']
x_data=[1,0,2,2,3]
y_data=[3,1,2,3,2]

one_hot_lookup=[[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1],]

x_one_hot=[one_hot_lookup[x] for x in x_data]
#print(x_one_hot)
#这一步x_one_hot变为独热编码一个列表，对应hello
inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels=torch.LongTensor(y_data).view(-1,1)

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super().__init__()
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.rnncell=torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

    def forward(self,input,hidden):
        hidden=self.rnncell(input,hidden)
        return hidden

    def init_hidden(self,):
        return torch.zeros(self.batch_size,self.hidden_size)

net=Model(input_size,hidden_size,batch_size)


criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.1)


for epoch in range(15):
    loss=0
    optimizer.zero_grad()
    hidden=net.init_hidden()
    print('predicted word:',end='')
    for input,label in zip(inputs,labels):
        hidden=net(input,hidden)
        loss+=criterion(hidden,label)
        _,idx =hidden.max(dim=1)
        print(idx2char[idx.item()],end='')
    loss.backward()
    optimizer.step()
    print(',Epoch[%d/15] loss=%.4f'%(epoch+1,loss.item()))

