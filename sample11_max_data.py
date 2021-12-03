import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
import copy
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from adabelief_pytorch import AdaBelief
device = torch.device('cuda')
def nmae(true,pred):
    score = np.mean(np.abs(true-pred) / true) * 100
    return score

def NMAE(true,pred):
    score = torch.mean(torch.abs(true-pred) / true) * 100
    return score

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.num_layers = 5
        self.hidden_size = 20
        self.lstm = nn.LSTM(20, 20, 5, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x_ = x[:,-1,:]
        out = out[:, -1, :] + x_
        out = self.relu(self.fc(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
class train_set(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, item):
        X = self.x[item]
        Y = self.y[item]
        return X,Y

class test_set(Dataset):
    def __init__(self,x):
        super().__init__()
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, item):
        X = self.x[item]
        return X

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_size = 5
        self.num_classes = 5
        self.linear1 = nn.Linear(self.hidden_size, 32)
        self.linear2 = nn.Linear(32, 32)
        self.fc = nn.Linear(32, self.num_classes)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.linear1(x))
        # print('1',out.shape)
        out = self.relu(self.linear2(out))
        # print('2',out.shape)
        out = self.fc(out)
        # print('3',out.shape)
        return out

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path,list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))

# start_date = '20210104'
start_date = '20160104'
end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']

for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    # model = LinearRegression()
    for code in tqdm(stock_list['종목코드'].values):
        data = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
        data = pd.merge(Business_days, data, how='outer')
        # data = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
        # print(data)
        # data = pd.merge(Business_days, data,how='left_on')
        data['weekday'] = data.Date.apply(lambda x: x.weekday())
        data['weeknum'] = data.Date.apply(lambda x: x.strftime('%V'))
        data.Close = data.Close.ffill()
        data.Open = data.Open.ffill()
        data.High = data.High.ffill()
        data.Low = data.Low.ffill()

        data['weeknum'] = data['weeknum'].map(lambda x: int(x))
        # print(data)
        # for i in data.values:
        #     print(i)
    #     break
    # break
        data2016 = data.query("Date >= '2016-01-04' and Date <= '2016-12-30'")
        data2017 = data.query("Date >= '2017-01-02' and Date <= '2017-12-29'")
        data2018 = data.query("Date >= '2018-01-01' and Date <= '2018-12-28'")
        data2019 = data.query("Date >= '2019-01-07' and Date <= '2019-12-27'")
        data2020 = data.query("Date >= '2019-12-30' and Date <= '2021-01-01'")
        data2021 = data.query("Date >= '2021-01-04'")

        wd2016 = data2016['weeknum'].iloc[-1]
        wd2017 = data2017['weeknum'].iloc[-1]
        wd2018 = data2018['weeknum'].iloc[-1]
        wd2019 = data2019['weeknum'].iloc[-1]
        wd2020 = data2020['weeknum'].iloc[-1]
        wd2021 = data2021['weeknum'].iloc[-1]

        data2017['weeknum'] = data2017['weeknum'].map(lambda x: x + wd2016)
        data2018['weeknum'] = data2018['weeknum'].map(lambda x: x + wd2016 + wd2017)
        data2019['weeknum'] = data2019['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018)
        data2020['weeknum'] = data2020['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018+wd2019)
        data2021['weeknum'] = data2021['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018+wd2019+wd2020)

        data = pd.concat([data2016,data2017,data2018,data2019,data2020,data2021],axis=0)

        data = pd.pivot_table(data=data, values=['Open','High','Low','Close'], columns='weekday', index='weeknum')
        data = data.dropna(axis=0)

        test_submission.loc[:, code] = pd.concat([data.iloc[-1], data.iloc[-1]], ignore_index=True)
        x = data.iloc[0:-1].to_numpy()  # 2021년 1월 04일 ~ 2021년 10월 22일까지의 데이터로
        c = x[-1]

        x_data = []
        y_data = []

        for r in range(1000000):
            if r+1 == len(x):
                break
            x_data.append(x[r:r+1])
            y_data.append(x[r+1,:5])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_public = data.iloc[-1].to_numpy().reshape(1,1,20)  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.

        model = RNN().to(device)
        opt = AdaBelief(model.parameters(),lr=1e-2,print_change_log=False,weight_decouple=False,rectify=False,degenerated_to_sgd=False)
        predictions = []
        testset = test_set(x_public)
        test_loader = DataLoader(testset,batch_size=256)
        # for y_value in y_values:
        trainset = train_set(x_data,y_data)
        trainloader = DataLoader(trainset, batch_size=256)
        for epoch in range(100):
            for i, (data,target) in enumerate(trainloader):
                opt.zero_grad()

                pred=model(data.float().to(device))
                # print(pred.shape)
                # print(target.shape)
                loss = NMAE(target.float().to(device),pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()

        pred_list = []
        for data in test_loader:

            pred = model(data.float().to(device))
            # print(pred.shape)
            pred_list.append(pred.detach().cpu().numpy())
        pred = np.concatenate(pred_list,axis=1)
        # print(pred.shape)
        predictions.append(pred)
        # print(sample_submission.loc[:, code].shape)
        # print(np.concatenate(predictions * 2,axis=1).shape)
        sample_submission.loc[:, code] = np.concatenate(predictions * 2, axis=1).reshape(-1)
    sample_submission.isna().sum().sum()

    columns = list(sample_submission.columns[1:])

    columns = ['Day'] + [str(x).zfill(6) for x in columns]

    sample_submission.columns = columns
    print(nmae(test_submission.iloc[:,1:].values,sample_submission.iloc[:,1:].values))


'''
100%|██████████| 370/370 [00:37<00:00,  9.81it/s]
4.989050763301384
100%|██████████| 370/370 [00:37<00:00,  9.99it/s]
4.70803904160944
100%|██████████| 370/370 [00:36<00:00, 10.03it/s]
3.6021737787866193
100%|██████████| 370/370 [00:36<00:00, 10.09it/s]
3.502820628405707
100%|██████████| 370/370 [00:37<00:00,  9.88it/s]
3.2457058532840546
100%|██████████| 370/370 [00:36<00:00, 10.15it/s]
3.46082442528933
100%|██████████| 370/370 [00:36<00:00, 10.14it/s]
4.279238224616672
'''

'''
D:\python_folder\python.exe D:/Administrator/dacon_binance/sample6.py
100%|██████████| 370/370 [03:52<00:00,  1.59it/s]
2.6803328776149526
100%|██████████| 370/370 [03:39<00:00,  1.69it/s]
2.76017541297922
100%|██████████| 370/370 [03:44<00:00,  1.65it/s]
2.759942682674028
100%|██████████| 370/370 [03:38<00:00,  1.69it/s]
2.7498304631037898
100%|██████████| 370/370 [03:40<00:00,  1.68it/s]
2.7370089111390006
100%|██████████| 370/370 [03:38<00:00,  1.69it/s]
2.8035905880900023
100%|██████████| 370/370 [03:39<00:00,  1.69it/s]
2.650328042627184
'''