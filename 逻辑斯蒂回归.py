import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
# train_set=torchvision.datasets.MNIST(root='./dataset/mnist',train=True,download=False)
# test_set=torchvision.datasets.MNIST(root='./dataset/mnist',train=False,download=False)

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred

model =LogisticRegressionModel()

criterion= torch.nn.BCELoss(size_average=False)
optimizer= torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(3000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# x_test=torch.Tensor([[4.0]])
# y_test=model(x_test)
# print('y_pred = ', y_test.data)

x=np.linspace(0,10,200)
x_t=torch.Tensor(x).view((200,1))
y_t=model(x_t)
y=y_t.data.numpy()

plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()

 
