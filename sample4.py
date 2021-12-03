import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
import copy
from sklearn.linear_model import LinearRegression,ElasticNet
from tqdm import tqdm
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from adabelief_pytorch import AdaBelief
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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

device = torch.device('cuda')
num_classes = 5
hidden_size = 1
num_layers = 5

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        # out = out[:,-1,:]
        # out = out.view(input_size,hidden_size)

        out = self.relu(self.fc(out[:, -1, :]))
        out = self.fc(out)
        out = self.fc2(out)
        # out = self.fc(out)
        return out

class ANN(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(ANN, self).__init__()
        self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        inner = 32
        self.linear1 = nn.Linear(hidden_size, inner)
        self.linear1 = nn.Linear(inner, inner)
        self.fc = nn.Linear(inner, num_classes)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.linear1(x))
        out = self.relu(self.linear2(out))
        out = self.relu(self.fc(out))
        return out

def nmae(true,pred):
    score = np.mean(np.abs(true-pred) / true) * 100
    return score

def NMAE(true,pred):
    score = torch.mean(torch.abs(true-pred) / true) * 100
    return score

path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path,list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))

start_date = '20210104'
end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']

validation_data = 1

for end_date in end_data_list:

    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    total_data = []
    code_list = []

    for code in tqdm(stock_list['종목코드'].values):
        data = fdr.DataReader(code, start=start_date, end=end_date)[['Close']].reset_index()
        data = pd.merge(Business_days, data, how='outer')
        data['weekday'] = data.Date.apply(lambda x: x.weekday())
        data['weeknum'] = data.Date.apply(lambda x: x.strftime('%V'))
        data.Close = data.Close.ffill()
        # print(data)
        data = pd.pivot_table(data=data, values='Close', columns='weekday', index='weeknum')
        # data = data.dropna(axis=0)
        total_data.append(data.values)
        code_list.append(code)
    data = np.stack(total_data,axis=1)

    x = data[0:-2]
    y = data[1:-1]

    y_0 = y[:, :,0]
    y_1 = y[:, :,1]
    y_2 = y[:, :,2]
    y_3 = y[:, :,3]
    y_4 = y[:, :,4]

    y_values = [y_0, y_1, y_2, y_3, y_4]

    for y_value in y_values:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y_value, train_size=0.8)
        train = train_set(x_train,y_train)
        valid = train_set(x_valid,y_valid)
        train_loader = DataLoader(train)
        valid_loader = DataLoader(valid)
        model = RNN(5, 370, 1).to(device)
        opt = AdaBelief(model.parameters(), lr=1e-5, print_change_log=False, weight_decouple=False, rectify=False,
                        degenerated_to_sgd=False)

        print(validation_data)
        validation_data +=1
        for epoch in range(50):
            train_loss = 0.
            val_loss = 0.
            for idx, (x_id, y_id) in enumerate(train_loader):
                opt.zero_grad()
                inputs = x_id.float().to(device)
                targets = y_id.float().to(device)
                pred = model(inputs)
                loss = NMAE(pred,targets)
                train_loss += loss
                loss.backward()
                opt.step()
            print("epoch",epoch,"train_loss", train_loss)
            for x_id,y_id in valid_loader:
                inputs = x_id.float().to(device)
                targets = y_id.float().to(device)
                pred = model(inputs)
                loss = NMAE(pred, targets)
                val_loss += loss
            print("epoch",epoch,"val_loss",val_loss)
    #     test_submission.loc[:, code] = pd.concat([data.iloc[-1], data.iloc[-1]], ignore_index=True)
    #     x = data.iloc[0:-2].to_numpy()  # 2021년 1월 04일 ~ 2021년 10월 22일까지의 데이터로
    #     y = data.iloc[1:-1].to_numpy()  # 2021년 1월 11일 ~ 2021년 10월 29일까지의 데이터를 학습한다.
    #     y_0 = y[:, 0]
    #     y_1 = y[:, 1]
    #     y_2 = y[:, 2]
    #     y_3 = y[:, 3]
    #     y_4 = y[:, 4]
    #
    #     y_values = [y_0, y_1, y_2, y_3, y_4]
    #     x_public = data.iloc[-2].to_numpy()  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.
    #
    #     predictions = []
    #     for y_value in y_values:
    #         x_train, x_valid, y_train,y_valid= train_test_split(x,y_value,train_size=0.8)
    #         train_loader = DataLoader(x_train,y_train)
    #         valid_loader = DataLoader(x_valid,y_valid)
    #         for idx , (x_id, y_id) in enumerate(train_loader):
    #             inputs = x_id.to(device)
    #             targets = y_id.to(device)
    #
    #         prediction = model.predict(np.expand_dims(x_public, 0))
    #         predictions.append(prediction[0])
    #     sample_submission.loc[:, code] = predictions * 2
    # sample_submission.isna().sum().sum()
    #
    # columns = list(sample_submission.columns[1:])
    #
    # columns = ['Day'] + [str(x).zfill(6) for x in columns]
    #
    # sample_submission.columns = columns
    # print(nmae(test_submission.iloc[:,1:].values,sample_submission.iloc[:,1:].values))

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