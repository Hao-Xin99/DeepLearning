####利用RNN
'''
import torch

batch_size=1
seq_len=3
input_size=4
hidden_size=2
num_layer=1


cell=torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layer)

input=torch.randn(seq_len,batch_size,input_size)
hidden=torch.zeros(num_layer,batch_size,hidden_size)

output,hidden=cell(input,hidden)

print('Out Size:',output.shape)
print('out:',output)
print('Hidden Size:',hidden.shape)
print('Hidden:',hidden)

'''

#字母序列重排
import torch

input_size=4  #输入特征数目，每一个时间步是（1，4）的，所以特征为4,也就是一共4个类型字母
hidden_size=4
batch_size=1

num_layers=1
seq_len=5




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
inputs=torch.Tensor(x_one_hot).view(seq_len,batch_size,input_size)
labels=torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super().__init__()
        self.num_layers=num_layers
        self.batch_size=batch_size
        self.hidden_size=hidden_size
        self.input_size=input_size
        self.rnn=torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

    def forward(self,input):
        hidden=torch.zeros(self.num_layers,self.batch_size,self.hidden_size)
        out,_=self.rnn(input,hidden)
        return out.view(-1,self.hidden_size)

net=Model(input_size,hidden_size,batch_size,num_layers)


criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.1)


for epoch in range(15):
    optimizer.zero_grad()
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    _,idx=outputs.max(dim=1)
    idx=idx.data.numpy()
    print('Predicted:',''.join([idx2char[x]for x in idx]),end='')
    print(',Epoch[%d/15] loss=%.3f'%(epoch+1,loss.item()))