from matplotlib import pyplot as plt

w=1.0
x_data =[1.0,2.0,3.0]
y_data =[2.0,4.0,6.0]
a=0.01

def forward(x):
    return w*x

def loss(x,y):
    return (forward(x)-y)**2

def gradient(x,y):
    return 2*x*(w*x-y)

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        LOSS = loss(x,y)
        w-=a*gradient(x,y)
    print("jindu:",epoch,"w:",w,"loss:",LOSS)


print("yuqi:",forward(4.0))