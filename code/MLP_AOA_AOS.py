# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 21:18:18 2020

@author: Administrator
"""

#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch
import torch.nn.functional as F
import time
import os

os.chdir(r'D:\research\aoa-s')
print("file open.")
#%%
train_data = pd.read_excel('./training_10.xlsx', index_col=0) # (441, 8)
print("train_data read over.")
test_data = pd.read_excel('./testing_10.xlsx', index_col=0) # (184+16, 8)
print("test_data read over.")
features_preprocess = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

# preprocessing
ss_x = preprocessing.StandardScaler()
train_data[features_preprocess] = ss_x.fit_transform(train_data[features_preprocess])
test_data[features_preprocess] = ss_x.transform(test_data[features_preprocess])

train_D14 = train_data[['D1', 'D4']] # (PV1)
train_D25 = train_data[['D2', 'D5']] # (PV2)
train_D36 = train_data[['D3', 'D6']] # (PV3)
train_MIX = train_data[['D1', 'D2', 'D3', 'D4', 'D5', 'D6']] # (PV1-3)
y_train = train_data[['AOA', 'AOS']]

test_D14 = test_data[['D1', 'D4']] # (PV1)
test_D25 = test_data[['D2', 'D5']] # (PV2)
test_D36 = test_data[['D3', 'D6']] # (PV3)
test_MIX = test_data[['D1', 'D2', 'D3', 'D4', 'D5', 'D6']] # (PV1-3)
y_test = test_data[['AOA', 'AOS']]

y_train = torch.tensor(y_train.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)


class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(n_input, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, n_output)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

start = time.time()

USE_DATA = 'MIX'
print(USE_DATA + ' start')
X_train, X_test = eval('train_'+USE_DATA), eval('test_'+USE_DATA)
X_train = torch.tensor(X_train.values, dtype=torch.float)
X_test = torch.tensor(X_test.values, dtype=torch.float)

BATCH_SIZE = 441
EPOCH = 2500

learning_rate = 0.001
weight_decay = 0.001

net = Net(X_train.shape[1], y_train.shape[1])

loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.01, weight_decay=weight_decay)
dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_iter = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                            shuffle=True) #, num_workers=2)

train_ls, test_ls = [], []
for epoch in range(EPOCH):
    for X, y in train_iter:
        y_pred = net(X)
        l = loss_function(y_pred, y.float())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

    test_l = loss_function(net(X_test), y_test).item()
    train_ls.append(l.item())
    test_ls.append(test_l)
    
    if epoch % 50 == 0:
        #test_l = loss_function(net(X_test), y_test).item()
        print(epoch, l.item(), test_l)
        #train_ls.append(l.item())
        #test_l = loss_function(net(X_test), y_test).item()
        #test_ls.append(test_l)

print('Training finished')

#%%
pred = net(X_test)
df = pd.DataFrame(pred.detach().numpy(), 
                    columns=[['AOA_pred(DEG)', 'AOS_pred(DEG)']], index=range(1, 201))
df[['AOA(DEG)', 'AOS(DEG)']] = test_data[['AOA', 'AOS']]
df['AOA_dis'] = np.abs(df[['AOA(DEG)']].values - df[['AOA_pred(DEG)']].values)
df['AOS_dis'] = np.abs(df[['AOS(DEG)']].values - df[['AOS_pred(DEG)']].values)

print(df)
aoaerror = float(df[['AOA_dis']].mean().values)
aoserror = float(df[['AOS_dis']].mean().values)
df2 = pd.DataFrame([{'AOA_error': df[['AOA_dis']].mean().values,
                        'AOS_error': df[['AOS_dis']].mean().values, }])

df3 = pd.DataFrame()
df3['train mse'] = train_ls
df3['test mse'] = test_ls

with pd.ExcelWriter('./' + USE_DATA + '/' + USE_DATA + '_A' + str('%.2f' % aoaerror) + '_S' + str('%.2f' % aoserror) + '.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet_1')
    df2.to_excel(writer, sheet_name='Sheet_2')
    df3.to_excel(writer, sheet_name='Sheet_3')

test_aoa = test_data['AOA'].values.tolist()
test_aos = test_data['AOS'].values.tolist()

pred_aoa = []
for i in df['AOA_pred(DEG)'].values.tolist():
    pred_aoa.append(i[0])

pred_aos = []
for i in df['AOS_pred(DEG)'].values.tolist():
    pred_aos.append(i[0])

testdata = [list(t) for t in zip(test_aoa, test_aos)]
preddata = [list(t) for t in zip(pred_aoa, pred_aos)]

#%% loss
# plt.figure(figsize=(8, 6))
# plt.plot(test_ls, label='test', c='r')
# plt.plot(train_ls, label='train', c='k', alpha=0.6)
# plt.legend()
# plt.grid()
# plt.ylim(0, 50)
# plt.title(USE_DATA+' loss curve')
# plt.text(200, 40, 'AOA mean error:{:.2f}'.format(float(df[['AOA_dis']].mean().values)))
# plt.text(200, 35, 'AOS mean error:{:.2f}'.format(float(df[['AOS_dis']].mean().values)))
# plt.xlabel('num of epoch')
# plt.ylabel('MSE loss')
# plt.savefig(USE_DATA+' loss curve.jpg')
# plt.show()

# %% pred
plt.figure(figsize=(8, 8))
plt.ylim(-23, 23)
plt.xlim(-23, 23)
plt.scatter(test_aos, test_aoa, marker='o', c='none', edgecolors='gray', label='Testing points')
plt.scatter(pred_aos, pred_aoa, marker='.', c='r', label='Prediction')
for i in range(len(testdata)):
    plt.plot([testdata[i][1], preddata[i][1]], [testdata[i][0], preddata[i][0]], color='r')

plt.legend(ncol=2, loc='upper right')
plt.grid()

plt.title('Prediction of testing points with ' + USE_DATA + ' model')
plt.text(-22, 22, 'AOA mean error:{:.2f}°'.format(float(df[['AOA_dis']].mean().values)))
plt.text(-22, 20.5, 'AOS mean error:{:.2f}°'.format(float(df[['AOS_dis']].mean().values)))
plt.xlabel('AOS(DEG)')
plt.ylabel('AOA(DEG)')
plt.savefig('./' + USE_DATA + '/' + USE_DATA + '_A'+str('%.2f' % aoaerror)+'_S'+str('%.2f' % aoserror)+'.svg', format='svg')
plt.show()

print(USE_DATA + ' over')
PATH = './' + USE_DATA + '/' + USE_DATA + '_A'+str('%.2f' % aoaerror)+'_S'+str('%.2f' % aoserror)+'.pt'
torch.save(net, PATH)

finish = time.time()
print('use time ', finish-start)