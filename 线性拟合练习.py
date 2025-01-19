import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#设函数为y=4x+2
x_data=[1.0,2.0,3.0]#原始数据
y_data=[6.0,10.0,14.0]

def forward(x):
    return x * w + b

def loss(x,y):
    y_pred=forward(x)
    return (y_pred-y)*(y_pred-y)

mse_list=[]
W=np.arange(0.0,4.1,0.1)
B=np.arange(0.0,4.1,0.1)

[w,b]=np.meshgrid(W,B)

l_sum=0
for x_val,y_val in zip(x_data,y_data):
    y_pre_val=forward(x_val)
    print(y_pre_val)
    loss_val=loss(x_val,y_val)
    l_sum+=loss_val

fig=plt.figure()
ax=Axes3D(fig)
ax.plot_surface(w,b,l_sum/3)
plt.show()
