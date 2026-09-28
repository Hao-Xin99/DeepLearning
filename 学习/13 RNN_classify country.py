##库加载
import torch
import time
import math
import matplotlib.pyplot as plt
import numpy as np
##参数准备
HIDDEN_SIZE=100
BATCH_SIZE=256
N_LAYER=2
N_EPOCH=100
N_CHARS=128
USE_GPU=True
#加载gpu
device = torch.device("cuda:0")
##预备函数
def create_tensor(tensor):    #决定在cpu,还是在gpu
    if USE_GPU:
        tensor=tensor.to(device)
    return tensor

def name2list(name):  #返回的是一个元组(arr,len),一个是转化后的数字列表,长度
    arr=[ord(c) for c in name]
    return arr,len(arr)

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

####数据加载
from torch.utils.data import DataLoader,Dataset #用来加载数据并分组
#import numpy as np  (np用来读取csv表格文件,不行,因为数据是字符
# `np.loadtxt` 是读数字矩阵的工具，不是读文本表格的工具。) 
# 处理文本的正确工具是 Python 的 `csv` 模块（或者 pandas)
import csv

class NameDataset(Dataset):
    def __init__(self,is_train_set=True):
        filename='./学习/数据集/13_name/names_train.csv' if is_train_set else './学习/数据集/13_name/names_test.csv'
        #csv读取文件
        with open(filename,encoding='utf8') as f:
            rows=list(csv.reader(f))
        #rows[:, 0]是 numpy/tensor语法，`csv` 读出来的是 Python 列表，不支持，要用列表推导 `[row[0] for row in rows]`,也不能转为numpy因为是字符
        #self.names=torch.from_numpy(readers[:,0])
        #self.countries=torch.from_numpy(readers[:,1])
        self.names=[row[0] for row in rows]
        self.countries=[row[1] for row in rows]
        self.len=len(self.names)
        self.country_list=list(sorted(set(self.countries)))
        #set()函数,接收可迭代对象,去重,返回集合{,,,}  ;  sorted()函数接收可迭代对象进行排序,按字母顺序升序排(A,B,C),返回列表
        self.country_dict=self.getCountryDict()
        self.country_num=len(self.country_list)

    def __getitem__(self, index):
        return self.names[index],self.country_dict[self.countries[index]]


    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dic=dict()
        for idx,c_name in enumerate(self.country_list):
            country_dic[c_name]=idx
        return country_dic
    
    def id2country(self,idx):
        return self.country_list[idx]

    def getCountryNum(self):
        return self.country_num

#实例化
train_set=NameDataset(is_train_set=True)
train_Loader=DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_set=NameDataset(is_train_set=False)
test_Loader=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False)

N_COUNTRY=train_set.getCountryNum()

#### 模型构建
class RNNClassifier(torch.nn.Module):
    def __init__(self,input_size,hidden_size,output_size,n_layers=1,bidirectional=True):
        super().__init__()
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.n_directions=2 if bidirectional else 1

        self.embedding=torch.nn.Embedding(input_size,hidden_size)
        self.gru=torch.nn.GRU(hidden_size,hidden_size,n_layers,bidirectional=bidirectional)

        self.fc=torch.nn.Linear(hidden_size*self.n_directions,output_size)

    def init_hidden(self,batch_size):
        hidden=torch.zeros(self.n_layers*self.n_directions,batch_size,self.hidden_size)
        return create_tensor(hidden) #加载到gpu

    def forward(self,input,seq_len):
        #输入的是ASCII码,padding,按长短排序后的姓名列表,本来是(batch,seq),转为(seq,batch)
        #把input转为(S,B),是因为RNN的输入要求的形状就是(S,B),不是batch_first
        input=input.t()
        batch_size=input.shape[1]
        #初始hidden,并把输入进行嵌入
        hidden=self.init_hidden(batch_size)
        after_embedding=self.embedding(input)

        from torch.nn.utils.rnn import pack_padded_sequence
        #这个函数的作用就是把这些已经按长短排好的数据,根据长度列表,按照序列次序和批数依次取,加快gru计算,
        #seq_len是input每个数据长度的列表
        gru_input=pack_padded_sequence(after_embedding,seq_len.to('cpu'))

        output,hidden=self.gru(gru_input,hidden)

        #把hidden拼接,而且原来的hidden是(个数,batch,hidden_size)三阶的
        #经过这一步变为(batch,hidden_size)二阶的
        if self.n_directions ==2:
            hidden_cat=torch.cat([hidden[-1],hidden[-2]],dim=1)#cat的tensor参数必须在[]里,表示把[]里的拼接
        else :
            hidden_cat=hidden[-1]

        fc_output=self.fc(hidden_cat)
        return fc_output

#模型实例化
classifier=RNNClassifier(input_size=N_CHARS,hidden_size=HIDDEN_SIZE,output_size=N_COUNTRY,n_layers=N_LAYER)
#损失函数和优化器
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(classifier.parameters(),lr=0.001)
    
def make_tensors(names,countries):  #把无序的名字数组,变为(batch,sequence)的张量,并填充0,排序
    seq_and_len=[name2list(name) for name in names ]
    name_sequences=[s[0] for s in seq_and_len]  #目前还是列表,不是张量
    seq_lengths=torch.LongTensor([s[1]for s in seq_and_len])
    countries=countries.long()

    #padding
    seq_tensor=torch.zeros(len(name_sequences),seq_lengths.max()).long()
    for idx,(seq,seq_len) in enumerate(zip(name_sequences,seq_lengths)):
        seq_tensor[idx,:seq_len]=torch.LongTensor(seq)

    #依据长度排序
    #第一句:sort会把seq_lengths按照降序排列,返回两个值,前者是排序后的张量,后者是索引张量,即第i个数据在原来的位置是j
    seq_lengths,perm_idx=seq_lengths.sort(dim=0,descending=True)  #descending,默认是False,升序,从小到大
    #第而三句:numpy/tensor的花式索引,索引的[]放一个整数张量,就能按张量里的值一次取出对应值,返回一个新张量
    seq_tensor=seq_tensor[perm_idx]
    countries=countries[perm_idx]

    return create_tensor(seq_tensor),\
           create_tensor(seq_lengths),\
           create_tensor(countries)

def trainModel():
    total_loss=0
    for i,(names,counties) in enumerate(train_Loader,1):
        inputs,seq_lens,target=make_tensors(names,counties)
        output=classifier(inputs,seq_lens)
        loss=criterion(output,target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss+=loss.item()
        if i%10==0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(train_set)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss 

def testModel():
    correct = 0
    total = len(test_set)
    print("evaluating trained model ...")
    with torch.no_grad():
        for _, (names, countries) in enumerate(test_Loader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


if __name__ == '__main__':
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    start = time.time()
    print("Training for %d epochs..." % N_EPOCH)
    acc_list = []
    for epoch in range(1, N_EPOCH + 1):
        # Train cycle
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

