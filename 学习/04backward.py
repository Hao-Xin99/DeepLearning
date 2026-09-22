import torch

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.Tensor([1.0])
w.requires_grad=True

def forward(x,w):
    return w*x

def loss(x,y):
    y_pred=forward(x,w)
    return (y-y_pred)**2

print("predict (before training)",4,forward(4,w).item())

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward()
        print("\t grad:",x,y,w.grad.item())
        w.data=w.data-0.01*w.grad.data
        #一定要注意tensor变量是否允许计算梯度,防止自行构建计算图,计算图也占用内存

        #梯度要清零,否则会累加
        w.grad.data.zero_()
    print("progress:",epoch,l.item())

print("predict (after training)",4,forward(4,w).item())



