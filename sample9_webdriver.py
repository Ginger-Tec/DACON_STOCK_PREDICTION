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
        self.num_layers = 1
        self.hidden_size = 5
        self.lstm = nn.LSTM(5, 5, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(5, 32)
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
start_date = '20170101'
end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    # model = LinearRegression()
    for stock in tqdm(stock_list.values):
        name = stock[0]
        code = stock[1]
        url = 'https://kr.investing.com/'
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text, 'html.parser')
        print(soup)
        break
    break
#         data = fdr.DataReader(code, start=start_date, end=end_date)[['Close']].reset_index()
#         data = pd.merge(Business_days, data, how='outer')
#         # data = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
#         # print(data)
#         # data = pd.merge(Business_days, data,how='left_on')
#         data['weekday'] = data.Date.apply(lambda x: x.weekday())
#         data['weeknum'] = data.Date.apply(lambda x: x.strftime('%V'))
#         data.Close = data.Close.ffill()
#         data = pd.pivot_table(data=data, values='Close', columns='weekday', index='weeknum')
#         data = data.dropna(axis=0)