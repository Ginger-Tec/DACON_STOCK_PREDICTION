import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
import copy
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import xgboost as xgb
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
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
import pywt
path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))

start_date = '20210104'
# start_date = '20160104'
# start_date = '20140104'
end_data_list = ['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']
# end_data_list = ['20211029','20211105','20211112']
import catboost as cat


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

def training(close, Model):
    collect = []
    label = []
    for yi in range(1, 2):
        x_close = close[:-5 * yi]
        model =None
        if Model == 1:
            model = ARIMA(x_close, order=(1, 1, 0))
        if Model == 2:
            model = ARIMA(x_close, order=(0, 1, 1))
        if Model == 3:
            model = SimpleExpSmoothing(x_close, initialization_method="estimated")
        model_fit = model.fit()
        forecast_close = model_fit.forecast(5)  # 마지막 5일의 예측 데이터
        collect.append(forecast_close)
        if yi == 1:
            label.append(close[-5 * yi:])
        else:
            label.append(close[-5 * yi:-5 * (yi - 1)])
    lin_data = np.stack(collect, axis=0)
    lin_label = np.stack(label, axis=0)
    return lin_data, lin_label

def testing(close, Model):
    model =None
    if Model == 1:
        model = ARIMA(close, order=(1, 1, 0))
    if Model == 2:
        model = ARIMA(close, order=(0, 1, 1))
    if Model == 3:
        model = SimpleExpSmoothing(close, initialization_method="estimated")
    model_fit = model.fit()
    forecast_close = model_fit.forecast(5)  # 마지막 5일의 예측 데이터
    return forecast_close

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
        self.num_layers = 2
        self.hidden_size = 2
        self.num_classes = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,)
        self.dense = nn.Linear(self.hidden_size,self.num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        out = self.sigmoid(self.dense(out))
        # x = x.view(-1,5)
        # out = x+out
        # out = self.dense(out)
        return out

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden_size = 5
        self.num_classes = 1
        self.linear1 = nn.Linear(self.hidden_size, self.num_classes)
        self.linear2 = nn.Linear(16, 16)
        self.fc = nn.Linear(16, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid =
    def forward(self,x):
        out = self.linear1(x)
        # # print('1',out.shape)
        # out = self.relu(self.linear2(out))
        # # print('2',out.shape)
        # out = self.relu(self.fc(out))
        # out = x + out
        # out = self.relu(self.fc2(out))
        # # print('3',out.shape)
        return out

for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    total_loss =0.
    for e,code in enumerate(tqdm(stock_list['종목코드'].values)):
        # if e==0:
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

            data = data.ffill()
            data = data.bfill()

            # print(data)
            x_public = data.iloc[-5:]

            x_public = x_public['Close']
            data = data.iloc[:-5]
            close = data['Close']

            collect = []
            label = []
            ar_data, ar_label = training(close,1)

            ar_data = ar_data.reshape(-1,5)

            label1 = ar_label[:, 0]
            label2 = ar_label[:, 1]
            label3 = ar_label[:, 2]
            label4 = ar_label[:, 3]
            label5 = ar_label[:, 4]
            # opt = AdaBelief(model.parameters(), lr=9e-1, print_change_log=False, weight_decouple=False, rectify=False,
            #                 degenerated_to_sgd=False)
            pred_list = []
            for label in [label1,label2,label3,label4,label5]:
                model = ANN().to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                trainset = train_set(ar_data, label.reshape(-1,1))
                trainloader = DataLoader(trainset, batch_size=16, pin_memory=True)
                for epoch in range(10):
                    # print(epoch)
                    train_loss = 0.
                    for i, load in enumerate(trainloader):
                        opt.zero_grad()
                        data1 = load['inputs1']
                        target = load['targets']
                        pred = model(data1.float().to(device, non_blocking=True))
                        loss = NMAE(target.float().to(device, non_blocking=True), pred)
                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        opt.step()
                        train_loss += loss.item()
                # print(train_loss)

                ar_test = testing(close, 1)
                ar_test = np.array(ar_test).reshape(1,5)
                testset = test_set(ar_test)
                test_loader = DataLoader(testset, batch_size=1)

                for load in test_loader:
                    data1 = load['inputs1']
                    pred = model(data1.float().to(device))
                    pred_list.append(pred.detach().cpu().numpy())
            pred = np.concatenate(pred_list).reshape(1,5)
            tmp = nmae(x_public, pred.reshape(-1))
            print(tmp)
            total_loss += tmp
    print(total_loss / 370)