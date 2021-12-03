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
    for e, code in enumerate(tqdm(stock_list['종목코드'].values)):
        if e==0:
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
            x = data['Close'].ewm(alpha=0.999).mean()
            test_submission.loc[:, code] = pd.concat([x[-5:], data.iloc[-5:]], ignore_index=True)
            train = x[:-5]
            model = ARIMA(train, order=(1, 1, 0))
            model_fit = model.fit()
            forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
            ori = x[-10:-5].values
            forecast_data = (forecast_data*0.75) + (ori*0.25)
            # print(nmae(x[-5:], forecast_data))
            tmp = nmae(x[-5:], forecast_data)
            total_loss += tmp
    print(total_loss/370)