import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

#数据加载
class TitanDataset(Dataset):
    def __init__(self,filepath):
        xy=pd.read_csv(filepath)
        self.len=xy.shape[0]
        #选取相关的数据特征
        feature=["Pclass","Sex","SibSp","Parch","Fare"]

        self.x_data = torch.from_numpy(np.array(pd.get_dummies(xy[feature])))
        self.y_data = torch.from_numpy(np.array(xy["Survived"]))

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    # 返回数据的条数/长度
    def __len__(self):
        return self.len


dataset = TitanDataset('dataset/titanic/train.csv')

train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        self.linear1=torch.nn.Linear(6,3)
        self.linear2=torch.nn.Linear(3,1)

        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        return x
    
    # 测试函数
    def test(self, x):
        with torch.no_grad():
            x=self.sigmoid(self.linear1(x))
            x=self.sigmoid(self.linear2(x))
            y=[]
            # 根据二分法原理，划分y的值
            for i in x:
                if i >0.5:
                    y.append(1)
                else:
                    y.append(0)
            return y




model=Model()
# 定义损失函数
criterion = torch.nn.BCELoss(reduction='mean')
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)


if __name__=='__main__':
    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
            x,y=data
            x=x.float()
            y=y.float()
            y_pred=model(x)
            y_pred=y_pred.squeeze(-1)
            loss=criterion(y_pred,y)
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 测试
test_data=pd.read_csv('dataset/titanic/test.csv')
feature = ["Pclass", "Sex", "SibSp", "Parch", "Fare"]
test=torch.from_numpy(np.array(pd.get_dummies(test_data[feature])))
y=model.test(test.float())
 
# 输出预测结果
output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':y})
output.to_csv('my_predict.csv',index=False)


