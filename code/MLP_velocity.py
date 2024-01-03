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

os.chdir(r'D:\research\UAVtest\MLP-VELO')
print("file open.")
#%%
train_data = pd.read_csv('./training.csv', index_col=False) # (43653, 8)
print("train_data read over.")
test_data = pd.read_csv('./testing_out.csv', index_col=False) # (1803+1851, 8)
print("test_data read over.")
features_preprocess = ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']

# preprocessing
ss_x = preprocessing.StandardScaler()
train_data[features_preprocess] = ss_x.fit_transform(train_data[features_preprocess])
test_data[features_preprocess] = ss_x.transform(test_data[features_preprocess])

train_MIX = train_data[['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']]
y_train = train_data[['VX', 'VY', 'VZ']]*1000

test_MIX = test_data[['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']]
y_test = test_data[['VX', 'VY', 'VZ']]*1000
y_train = torch.tensor(y_train.values, dtype=torch.float)
y_test = torch.tensor(y_test.values, dtype=torch.float)

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(n_input, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, n_output)
    
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

BATCH_SIZE = 4096
EPOCH = 50

learning_rate = 0.001
weight_decay = 0.0005

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
                    columns=[['X_pred()', 'Y_pred()', 'Z_pred()']], index=range(1, 1852))/1000 #1804+1852
df[['VX', 'VY', 'VZ']] = y_test/1000
df['X_dis'] = np.abs(df[['VX']].values - df[['X_pred()']].values)
df['Y_dis'] = np.abs(df[['VY']].values - df[['Y_pred()']].values)
df['Z_dis'] = np.abs(df[['VZ']].values - df[['Z_pred()']].values)

print(df)
Xerror = float(df[['X_dis']].mean().values)
Yerror = float(df[['Y_dis']].mean().values)
Zerror = float(df[['Z_dis']].mean().values)
df2 = pd.DataFrame([{'X_error': df[['X_dis']].mean().values,
                        'Y_error': df[['Y_dis']].mean().values,
                        'Z_error': df[['Z_dis']].mean().values}])

df3 = pd.DataFrame()
df3['train mse'] = train_ls
df3['test mse'] = test_ls

with pd.ExcelWriter(USE_DATA + '_X' + str('%.2f' % Xerror) + '_Y'+str('%.2f' % Yerror) + '_Z'+str('%.2f' % Zerror) + '.xlsx') as writer:
    df.to_excel(writer, sheet_name='Sheet_1')
    df2.to_excel(writer, sheet_name='Sheet_2')
    df3.to_excel(writer, sheet_name='Sheet_3')

test_X = test_data['VX'].values.tolist()
test_Y = test_data['VY'].values.tolist()
test_Z = test_data['VZ'].values.tolist()
pred_X = []
for i in df['X_pred()'].values.tolist():
    pred_X.append(i[0])

pred_Y = []
for i in df['Y_pred()'].values.tolist():
    pred_Y.append(i[0])
pred_Z = []
for i in df['Z_pred()'].values.tolist():
    pred_Z.append(i[0])

testdata = [list(t) for t in zip(test_X, test_Y, test_Z)]
preddata = [list(t) for t in zip(pred_X, pred_Y, pred_Z)]

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
test_T = test_data['Time'].values.tolist()
plt.figure(figsize=(18, 9), dpi=300)

plt.subplot(3, 1, 1)
plt.ylim(-2, 2)
# plt.xlim(33.8, 107.8)
plt.xlim(594.55, 631.55)
plt.scatter(test_T, test_X, marker='o', c='none', edgecolors='gray', label='Testing points')
plt.scatter(test_T, pred_X, marker='.', c='r', label='Prediction')
for i in range(len(testdata)):
    plt.plot([testdata[i][1], preddata[i][1]], [testdata[i][0], preddata[i][0]], color='r')
plt.legend(ncol=2, loc='upper right')
plt.grid()

plt.subplot(3, 1, 2)
plt.ylim(-4, 3)
# plt.xlim(33.8, 107.8)
plt.xlim(594.55, 631.55)
plt.scatter(test_T, test_Y, marker='o', c='none', edgecolors='gray', label='Testing points')
plt.scatter(test_T, pred_Y, marker='.', c='r', label='Prediction')
for i in range(len(testdata)):
    plt.plot([testdata[i][1], preddata[i][1]], [testdata[i][0], preddata[i][0]], color='r')
plt.legend(ncol=2, loc='upper right')
plt.grid()

plt.subplot(3, 1, 3)
plt.ylim(-3, 4.5)
# plt.xlim(33.8, 107.8)
plt.xlim(594.55, 631.55)
plt.scatter(test_T, test_Z, marker='o', c='none', edgecolors='gray', label='Testing points')
plt.scatter(test_T, pred_Z, marker='.', c='r', label='Prediction')
for i in range(len(testdata)):
    plt.plot([testdata[i][1], preddata[i][1]], [testdata[i][0], preddata[i][0]], color='r')
plt.legend(ncol=2, loc='upper right')
plt.grid()

plt.title('Prediction of testing points with ' + USE_DATA + ' model')
plt.text(-22, 22, 'AOA mean error:{:.2f}°'.format(float(df[['X_dis']].mean().values)))
plt.text(-22, 20.5, 'AOS mean error:{:.2f}°'.format(float(df[['Y_dis']].mean().values)))
plt.xlabel('AOS(DEG)')
plt.ylabel('AOA(DEG)')
plt.savefig(USE_DATA + '_X'+str('%.2f' % Xerror)+'_Y'+str('%.2f' % Yerror)+'_Z'+str('%.2f' % Zerror)+'.svg', format='svg')
plt.show()
PATH = USE_DATA + '_X'+str('%.2f' % Xerror)+'_Y'+str('%.2f' % Yerror)+'_Z'+str('%.2f' % Zerror)+'.pt'
torch.save(net, PATH)
print(USE_DATA + ' over')

finish = time.time()
print('use time ', finish-start)