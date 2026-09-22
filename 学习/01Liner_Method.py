import  numpy as np
from matplotlib import pyplot as plt

x_data =[1.0,2.0,3.0]
y_data =[2.0,4.0,6.0]


def forward(w,x):
    return w*x

def loss(w,x,y):#一个样本的损失
    y_pred=forward(w,x)
    return (y_pred-y)**2

w_jilu=[]
MSE_jilu=[]

for w in np.arange(0.0,4.1,0.1):
    sum=0
    print("w=",w)
    for x,y in zip(x_data,y_data):
        the_loss=loss(w,x,y)
        sum+=the_loss
        print(f"x={x},y={y},loss={the_loss}")
    mes=sum/3
    print(f"MSE={mes}")
    MSE_jilu.append(mes)
    w_jilu.append(w)

plt.plot(w_jilu,MSE_jilu)
plt.show()



