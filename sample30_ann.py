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
        self.hidden_size = 40
        self.num_classes = 5
        self.linear1 = nn.Linear(self.hidden_size, 64)
        self.linear2 = nn.Linear(64, 64)
        self.fc = nn.Linear(64, self.num_classes)
        # self.dropout = nn.Dropout(0.9)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.linear1(x))
        # print('1',out.shape)
        out = self.relu(self.linear2(out))
        # out = self.dropout(out)
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
start_date = '20160106'
end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']

for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    # model = LinearRegression()
    for code in tqdm(stock_list['종목코드'].values):
        data = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
        data['weekday'] = data.Date.apply(lambda x: x.weekday())
        data['weeknum'] = data.Date.apply(lambda x: x.strftime('%V'))
        data = data.drop(columns='Date')
        data.Close = data.Close.ffill()
        data.Open = data.Open.ffill()
        data.High = data.High.ffill()
        data.Low = data.Low.ffill()
        data.Volume = data.Volume.ffill()
        data.Change = data.Change.ffill()
        data.weekday = data.weekday.ffill()
        data.weeknum = data.weeknum.ffill()
        # break
        #'High','Low','Open','Volume','Change'
        data = pd.pivot_table(data=data, values=['Close','Open','High','Low'], columns='weekday', index='weeknum')
        data = data.dropna(axis=0)


        test_submission.loc[:, code] = pd.concat([data.iloc[-1], data.iloc[-1]], ignore_index=True)
        pre = data.iloc[0:-3].to_numpy()  # 2021년 1월 04일 ~ 2021년 10월 22일까지의 데이터로
        x = data.iloc[1:-2].to_numpy()
        x = np.concatenate([pre,x],axis=1)

        y = data.iloc[2:-1,0:5].to_numpy()  # 2021년 1월 11일 ~ 2021년 10월 29일까지의 데이터를 학습한다.
        x_public_pre = data.iloc[-3].to_numpy().reshape(1,-1)
        x_public = data.iloc[-2].to_numpy().reshape(1,-1)  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.
        x_public = np.concatenate([x_public_pre,x_public],axis=1)
        model =ANN().to(device)
        opt = AdaBelief(model.parameters(),lr=1e-3,print_change_log=False,weight_decouple=False,rectify=False)
        predictions = []
        testset = test_set(x_public)
        test_loader = DataLoader(testset,batch_size=32)
        # for y_value in y_values:
        trainset = train_set(x,y)
        trainloader = DataLoader(trainset, batch_size=32)
        for epoch in range(10):
            epoch_loss = 0.
            for i, (data,target) in enumerate(trainloader):
                opt.zero_grad()
                pred=model(data.float().to(device))
                # print(pred.shape)
                # print(target.shape)
                loss = NMAE(target.float().to(device),pred)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            # print(epoch,epoch_loss)
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