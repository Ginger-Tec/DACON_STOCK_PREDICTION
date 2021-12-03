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
        data = data.ffill()
        data = data.bfill()
        x = data['Close']
        test_submission.loc[:, code] = pd.concat([x[-5:], data.iloc[-5:]], ignore_index=True)
        train = x[:-5]
        # fit1 = SimpleExpSmoothing(train, initialization_method="estimated").fit()
        # fit1 = Holt(train, damped_trend=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.2)
        # fit1 = Holt(train, initialization_method="estimated").fit()
        # fit1 = Holt(train, exponential=True, initialization_method="estimated").fit()
        # fit1 = Holt(train, damped_trend=True, initialization_method="estimated").fit()
        fit1 = ExponentialSmoothing(
            train,
            seasonal_periods=4,
            trend="add",
            seasonal="add",
            # use_boxcox=True,
            initialization_method="estimated",
        ).fit()
        # fit1 = Holt(train, damped_trend=True, exponential=True, initialization_method="estimated").fit()
        forecast_data = fit1.forecast(5).rename(r"$\alpha=%s$" % fit1.model.params["smoothing_level"])
        # print(x[-5:])
        # print(forecast_data)
        # sample_submission.loc[:, code] = forecast_data.tolist() * 2
        # print(sample_submission.loc[:, code].values)
        total_loss += nmae(x[-5:], forecast_data)
    print(total_loss/370)

'''
100%|██████████| 370/370 [00:54<00:00,  6.79it/s]
4.187007660266795
100%|██████████| 370/370 [00:54<00:00,  6.79it/s]
3.4285093851128443
100%|██████████| 370/370 [00:53<00:00,  6.86it/s]
2.421393994569225
100%|██████████| 370/370 [00:54<00:00,  6.84it/s]
2.6466587881321524
100%|██████████| 370/370 [00:53<00:00,  6.87it/s]
2.791668159576136
100%|██████████| 370/370 [00:54<00:00,  6.85it/s]
2.957840417167324
100%|██████████| 370/370 [00:53<00:00,  6.90it/s]
3.6188382449000467
'''

'''
100%|██████████| 370/370 [01:03<00:00,  5.78it/s]
  0%|          | 0/370 [00:00<?, ?it/s]3.8946260745822445
100%|██████████| 370/370 [01:03<00:00,  5.83it/s]
3.529135974502169
100%|██████████| 370/370 [01:03<00:00,  5.81it/s]
3.5653276805995326
100%|██████████| 370/370 [01:03<00:00,  5.79it/s]
2.9402296396616165
100%|██████████| 370/370 [01:03<00:00,  5.79it/s]
3.1794021732789295
100%|██████████| 370/370 [01:03<00:00,  5.82it/s]
3.5239926941671356
100%|██████████| 370/370 [01:04<00:00,  5.75it/s]
3.872637078786939
'''
'''
100%|██████████| 370/370 [01:19<00:00,  4.66it/s]
4.198131278864746
100%|██████████| 370/370 [01:19<00:00,  4.68it/s]
3.473987417956368
100%|██████████| 370/370 [01:20<00:00,  4.61it/s]
2.6171877155167413
100%|██████████| 370/370 [01:20<00:00,  4.62it/s]
2.658906504128803
100%|██████████| 370/370 [01:19<00:00,  4.66it/s]
2.861108005598131
100%|██████████| 370/370 [01:20<00:00,  4.61it/s]
3.3842997592213715
100%|██████████| 370/370 [01:20<00:00,  4.60it/s]
3.5389229170403183
'''
'''
100%|██████████| 370/370 [01:21<00:00,  4.56it/s]
4.222821918569429
100%|██████████| 370/370 [01:21<00:00,  4.54it/s]
3.4297733071040493
100%|██████████| 370/370 [01:21<00:00,  4.52it/s]
2.5017731733044664
100%|██████████| 370/370 [01:21<00:00,  4.55it/s]
2.6408736555310686
100%|██████████| 370/370 [01:21<00:00,  4.54it/s]
2.7745203376597196
100%|██████████| 370/370 [01:21<00:00,  4.55it/s]
2.9646688937022745
100%|██████████| 370/370 [01:21<00:00,  4.56it/s]
3.5801186341191795
'''
'''
100%|██████████| 370/370 [03:18<00:00,  1.86it/s]
4.145601179800285
100%|██████████| 370/370 [03:18<00:00,  1.86it/s]
3.438100251260252
100%|██████████| 370/370 [03:16<00:00,  1.89it/s]
2.6821510456603885
100%|██████████| 370/370 [03:17<00:00,  1.88it/s]
2.649561890981625
100%|██████████| 370/370 [03:18<00:00,  1.86it/s]
2.8765665876266047
100%|██████████| 370/370 [03:18<00:00,  1.87it/s]
3.091449208564871
100%|██████████| 370/370 [03:20<00:00,  1.84it/s]
3.562763933418493

'''