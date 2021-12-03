import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
import copy
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import xgboost as xgb
import warnings

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
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
from xgboost import plot_importance
from matplotlib import pyplot


def nmae(true, pred):
    score = np.mean(np.abs(true - pred) / true) * 100
    return score
def NMAE(true, pred):
    score = torch.mean(torch.abs(true-pred) / true) * 100
    return score

from sklearn.metrics import mean_absolute_error

path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))

# start_date = '20210104'
start_date = '20160104'
end_data_list = ['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']
# end_data_list = ['20211029','20211105','20211112']
import catboost as cat
#dataset
class train_set(Dataset):
    def __init__(self, x1, y):
        super().__init__()
        self.x1 = x1
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, item):
        items = {}
        X1 = self.x1[item]
        Y = self.y[item]
        items['inputs1'] = X1
        items['targets'] = Y
        return items

class test_set(Dataset):
    def __init__(self, x1):
        super().__init__()
        self.x1 = x1
    def __len__(self):
        return len(self.x1)
    def __getitem__(self, item):
        items = {}
        X1 = self.x1[item]
        items['inputs1'] = X1
        return items
#model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.input_size=1
        self.num_layers = 1
        self.hidden_size = 16
        self.num_classes = 5
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.dense = nn.Linear(self.hidden_size,self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        out = self.dense(out)
        return out
#criterion




def Make_Data(data):
    keep = []
    for e, i in enumerate(data.values):
        if (i[-3] == 0) and (i[-2] == 1):
            keep.append(e)
        if e == len(data)-1:
            keep.append(e)

    date = []
    for k in keep:
        date.append(data.loc[k, 'Date'])
    df = []
    for d in range(len(date) - 1):
        if d==len(date)-2:
            df.append(data.query("Date >= '{}'".format(str(date[d]).split(' ')[0])))
        else:
            df.append(data.query("Date >= '{}' and Date < '{}'".format(str(date[d]).split(' ')[0], str(date[d+1]).split(' ')[0])))

    ord = []
    for d in df:
        ord.append(d['weeknum'].iloc[-1])

    ord1 = []
    base = 0
    for d in ord:
        base += d
        ord1.append(base)

    frame = pd.DataFrame()
    for d, b in zip(df, ord1):
        d['weeknum'] = d['weeknum'].map(lambda x: x + b)
        frame = pd.concat([frame, d], ignore_index=True)

    return frame


for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    model = LinearRegression()
    total_loss =0.
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
        data.Volume = data.Volume.ffill()
        data.Change = data.Change.ffill()
        data.weekday = data.weekday.ffill()
        data.weeknum = data.weeknum.ffill()
        # data['weekday'] = data['weekday'].map(lambda x: int(x))
        data['weeknum'] = data['weeknum'].map(lambda x: int(x))
        data['week'] = data.weeknum

        data=Make_Data(data)
        # data = data.dropna(method='')
        data = data.ffill()
        data = data.bfill()
        x = data['Close']
        test_submission.loc[:, code] = pd.concat([x[-5:], data.iloc[-5:]], ignore_index=True)
        train = np.log(x[:-5])
        x_train = []
        y_train = []
        for i in range(len(train)):
            if i+10 == len(train):
                break
            x_train.append(train[i:i+5])
            y_train.append(train[i+5:i+10])
        x_train = np.array(x_train).reshape(-1,5,1)
        y_train = np.array(y_train)

        x_public = np.array(x[-10:-5]).reshape(1,5,1)
        model = RNN().to(device)
        # opt = AdaBelief(model.parameters(), lr=9e-1, print_change_log=False, weight_decouple=False, rectify=False,
        #                 degenerated_to_sgd=False)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        predictions = []

        testset = test_set(x_public)
        test_loader = DataLoader(testset, batch_size=1)
        # for y_value in y_values:
        trainset = train_set(x_train, y_train)
        trainloader = DataLoader(trainset, batch_size=2048, pin_memory=True)
        for epoch in range(1000):
            print(epoch)
            train_loss = 0.
            for i, load in enumerate(trainloader):
                opt.zero_grad()
                data1 = load['inputs1']
                target = load['targets']
                print("target",target)
                pred = model(data1.float().to(device, non_blocking=True))
                print("pred",pred)
                loss = nn.L1Loss()(target.float().to(device, non_blocking=True), pred)
                acc = NMAE(target.float().to(device, non_blocking=True), pred)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                train_loss += acc.item()
            # print('train_loss: ',train_loss)


        pred_list = []
        for load in test_loader:
            data1 = load['inputs1']
            pred = model(data1.float().to(device))
            pred_list.append(pred.detach().cpu().numpy())
        pred = np.concatenate(pred_list, axis=1).reshape(5)
        print(x[-5:])
        print(pred)
        print(nmae(x[-5:],pred))
        # total_loss
