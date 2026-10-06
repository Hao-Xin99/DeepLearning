import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torch.nn.functional as F


####数据集加载  数据处理
class TitanicDataSet(Dataset):
    def __init__(self,is_train,age_median=None,age_mean=None,age_std=None,emb_mode=None,Fare_median=None,Fare_mean=None,Fare_std=None):
        self.is_train=is_train
        filepath='./学习/数据集/08_titanic/train.csv'if is_train else './学习/数据集/08_titanic/test.csv'
        #1.先把数据集读进来,看一下数据集的结构,并看一下缺失值
        self.df=pd.read_csv(filepath)  #用df.isnull().sum()求出来所有特征的缺失值,打印出来
        #2.确定是一个二分类,生死
        #3.对特征进行去留,分出来特征和标签
        x=self.df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]  #保留特征
            #里面把'Name','Tiket'还有'Cabin'丢了
            #主要是"Name"唯一性高,不好学特征,且是文本特征,不好处理.剩下两个文本数字混杂,不好处理,缺失值多.而且与预测主观上无关
        y=self.df['Survived'] if is_train else None  #标签,只有训练集有
        #4.对缺失值进行处理(必须用训练集的数据填补)
        if is_train:
            self.age_median=x['Age'].median()
            self.Fare_median=x['Fare'].median()
            self.emb_mode=x['Embarked'].mode()[0]
        else:
            self.age_median=age_median
            self.Fare_median=Fare_median
            self.emb_mode=emb_mode
        x['Age']=x['Age'].fillna(self.age_median) #中位数填补
        x['Fare']=x['Fare'].fillna(self.Fare_median) 
        x['Embarked']=x['Embarked'].fillna(self.emb_mode)

        #5. 对文本类的进行编码,有序012,无序one-hot
        #这里对sex和Embarked独热编码
        #sex是二分类,用一列01就能表示,无大小关系
        x['Sex']=x['Sex'].map({'male':1,'female':0})
        #Embarked是三分类,用独热,先把这一列拆分成3个特征,再合并到原来的
        embarked_dummies=pd.get_dummies(x['Embarked'],prefix='Emb')
        x=pd.concat([x.drop(columns=['Embarked']),embarked_dummies],axis=1)
        
        #6.标准化:把数字特征分布转化为标准正态分布(均值和方差也要用训练集的)
        if is_train:
            self.age_mean=x['Age'].mean()
            self.age_std =x['Age'].std()
            self.Fare_mean=x['Fare'].mean()
            self.Fare_std =x['Fare'].std()
        else:
            self.age_mean=age_mean
            self.age_std=age_std
            self.Fare_mean=Fare_mean
            self.Fare_std=Fare_std
        
        x['Age']=(x['Age']-self.age_mean)/self.age_std
        x['Fare']=(x['Fare']-self.Fare_mean)/self.Fare_std

        #7.不看行列标签,把里面的值全部转换为np数组,再转换为tensor张量
        #self.x=x
        self.x_data=x.values.astype(np.float32) #必须全转为float32
        self.x_data=torch.from_numpy(self.x_data)
        if is_train:
            self.y_data=y.values.astype(np.int64) #标签必须转为int64(long)
            self.y_data=torch.from_numpy(self.y_data)




    def __getitem__(self, index):
        if self.is_train:
            return self.x_data[index],self.y_data[index]
        else :
            return self.x_data[index]

    def __len__(self,):
        return len(self.x_data)

train_set=TitanicDataSet(is_train=True)
train_loader=DataLoader(train_set,batch_size=32,shuffle=True)

age_median=train_set.age_median
Fare_median=train_set.Fare_median
emb_mode=train_set.emb_mode
age_mean=train_set.age_mean
age_std=train_set.age_std
Fare_mean=train_set.Fare_mean
Fare_std=train_set.Fare_std

test_set=TitanicDataSet(
    is_train=False,age_median=age_median,
    age_mean=age_mean,age_std=age_std,
    emb_mode=emb_mode,Fare_median=Fare_median,
    Fare_mean=Fare_mean,Fare_std=Fare_std
)
test_loader=DataLoader(test_set,batch_size=32,shuffle=False)

#### 模型构建

class Titanic_Mode(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc1=torch.nn.Linear(9,16)
        self.fc2=torch.nn.Linear(16,4)
        self.fc3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()

        #torch.nn.functional.relu()  就是relu函数本身,可以直接用
        #torch.nn.ReLU是一个类,需要实例化,构造一个可以用的relu变量

    def forward(self,input):
        input=self.sigmoid(self.fc1(input))
        input=self.sigmoid(self.fc2(input))      
        input=self.sigmoid(self.fc3(input))  
        return input
        #最后一层
module=Titanic_Mode()

####损失函数与优化器
criterion=torch.nn.BCELoss()
optimizer=torch.optim.Adam(module.parameters(),lr=0.03)

####训练循环

def Train():
    for batch_idx,(x,y) in enumerate(train_loader): #batch_idx最多应该就891/32轮
        #梯度清零
        optimizer.zero_grad()
        #前馈,算损失
        y_pred=module(x)
        loss=criterion(y_pred,y.view(-1,1).float())
        #反馈
        loss.backward()
        #更新
        optimizer.step()

        #打印训练进度
        if batch_idx%10==0:
            print(f"Epoch:{epoch+1} batch_idx:{batch_idx} loss:{loss}")

def Test():
    the_answer=[]
    with torch.no_grad():
        for x in test_loader:  #每个x都是一批数据
            y_pred=module(x)
            answer=(y_pred>=0.5).float()
            the_answer.append(answer.squeeze(1))  #这样收集的还是一批批的列表[[第一次],[第二次]],要拼接
    predicated=torch.cat(the_answer)#返回的必须是一维数组,(32,)形状,而不是(32,1)形状
    return predicated.numpy().astype(np.int64)#
#训练并更新权重
for epoch in range(50):
    Train()

#把答案提交
pred=Test()
submit=pd.DataFrame({
    'PassengerId':test_set.df['PassengerId'],
    'Survived':pred
})
save_path='学习/08_2 titanic_answer_2.csv'
submit.to_csv(save_path,index=False)
    
print(submit.head())




