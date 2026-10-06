import pandas as pd
import torch

'''
##利用pd.Series(value,key)创建一维pandas对象
#Series是字典,要求输入键值对,由键可以找到值,value是值,可以是(列表,np数组,tensor张量).
v=[0.1,0.2,0.3]
k=['a','b','c']
v=torch.tensor(v)
print(v)
sr=pd.Series(v,k)
print(sr) 
print(sr.values)#输出值的形式都是用numpy数组的形式输出,无论value传入的时候是列表,数组还是张量
print(sr.index)#index,索引,也就是键


##创建二维pandas对象
value=[[13,'男'],[15,'女'],[16,'男']]
key=['1号','2号','3号']
columns=['年龄','性别']

df=pd.DataFrame(value,index=key,columns=columns)



##索引,分为显式和隐式索引
print(df['年龄'])  
print(df[['年龄','性别']])   #这两种直接加[],意思只有取一整列.  第二行的实质是花式索引[]里用列表表示取这两列

print(df.loc['1号','年龄'])  
print(df.loc[['1号','2号'],'年龄'])    #.loc[,]显式索引.[]里面有两个值,第一个表示行,第二个表示列.均可以用列表取多行多列(花样索引)
print(df.loc['1号':'2号','年龄'])  #1.显式的切片两边都闭  2.切片本身就是合法参数,不能单独放入列表


print(df.iloc[0,0])   #隐式索引同理,不同的是切片左闭右开
print(df.iloc[0:1,0])  #切片得到的输出是矩阵

'''


# pandas里可同时储存数字变量和字符变量

##实际导入csv文件
df=pd.read_csv('./学习/数据集/07_diabetes/diabetes test.csv',index_col=0,encoding='gb18030')
#print(df)


##查询缺失值
#print(df.isnull())

##去除缺失值(默认去除样本行)
#print(df.dropna())
#print(df.dropna(axis=1)) #去除列
#print(df.dropna(how=1)) #当nan个数超过1时去除


##填充缺失值,用.fillna()函数
#对于数字:直接填一个数字,就按那个数字填充
#print(df.fillna(0))
#import numpy as np
#print(df.fillna(np.mean(df)))#用df每一列的均值填充,使用np的函数

#对于字母,也是直接填
#print(df.fillna('你好'))

#df['指标3']=df['指标3'].fillna('你好')
#print(df)

##pandas里求均值，中位数，众数
mean=df.mean()
median=df.median()
zhong=df.mode()[0]
std=df.std()#求标准差,数值类的用
print(f'mean:{mean}  中位数：{median}  众数：{zhong}')