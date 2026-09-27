import torch
#超参数
num_class=4  # 最后分类的类别数
input_size=4
hidden_size=8
embedding_size=10
num_layers=2
batch_size=1
seq_len=5

#数据集准备
idx2char=['e','h','l','o']
x_data=[[1,0,2,2,3]]   #(batch_size,seq_len)
y_data=[3,1,2,3,2]   #(batch_size*seq_len)

inputs=torch.LongTensor(x_data)
labels=torch.LongTensor(y_data)

#模型构建
class Model(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.emb=torch.nn.Embedding(input_size,embedding_size)
        self.rnn=torch.nn.RNN(input_size=embedding_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)  
        self.fc=torch.nn.Linear(hidden_size,num_class)
        #  batch_first=Ture，是对于传进网络的x的形状来说的
    def forward(self,x):   #x是（batch_size,seq_len,input_size）的形状
        hidden=torch.zeros(num_layers,x.shape[0],hidden_size)
        x=self.emb(x)       #把x嵌入到稠密区域，变为（batch_size,seq_len,embedding_size）形状
        x,_=self.rnn(x,hidden)  #这一步x的形状就变为了（batch_size,seq_len,hidden_size）形状
        x=self.fc(x)        #线性层把x的最后一维进行变换，变为（batch_si    ze,seq_len,num_class）
        return x.view(-1,num_class) #变为矩阵，行数确定数量和顺序，列数是预测值

net=Model()

criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()

    output=net(inputs)
    loss=criterion(output,labels)
    loss.backward()
    optimizer.step()

    _,idx=output.max(dim=1)
    idx=idx.data.numpy()#变为np数组，不变其实也可以
    print('Predoicted:',''.join(idx2char[x] for x in idx),end='')
    print(',Epoch[%d/15] loss:%.3f'%(epoch+1,loss.item()))

