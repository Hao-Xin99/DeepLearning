from matplotlib import pyplot as plt

w=1.0
x_data =[1.0,2.0,3.0]
y_data =[2.0,4.0,6.0]
loss_jilu=[]
w_jilu=[]
def predict_y(x):
    return x*w

def cost(xs,ys):
    the_all=0
    for x,y in zip(xs,ys):
        y_pre=predict_y(x)
        the_all+=(y_pre-y)**2
    return the_all/len(xs)

def grad(xs,ys):
    total=0
    for x,y in zip(xs,ys):
        total+=2*(w*x-y)*x
    return total/len(xs)

for epoch  in range(0,100):
    the_loss=cost(x_data,y_data)
    loss_jilu.append(the_loss)
    grads=grad(x_data,y_data)
    w_jilu.append(w)   
    w-=0.01*grads
    print(f"loss:{the_loss:.4f}   grads:{grads:.4f}")

plt.plot(w_jilu,loss_jilu)
plt.show()




