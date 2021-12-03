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
from neuralprophet import NeuralProphet
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,RobustScaler
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


end_data_list =['20211105','20211126']
import catboost as cat

storage = {}
for i in range(370):
    storage[i] = 1
storage[113] = 2
storage[258] = 2
storage[212] = 2
storage[206] = 2
storage[31] = 3
storage[288] = 4
storage[133] = 4
storage[229] = 4
storage[295] = 4
storage[123] = 4
storage[325] = 5
storage[234] = 5
storage[333] = 5
storage[240] = 5
storage[345] = 5
storage[6] = 5
storage[322] = 6
storage[260] = 6
storage[46] = 6
nyear = [325, 6, 301, 18, 19, 28]
storage[210] = 9
storage[147] = 10
storage[116] = 10
storage[310] = 10
storage[299] = 11
storage[44] = 11
storage[269] = 11
storage[150]=11

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

sample_submission = pd.read_csv(os.path.join(path, sample_name))
test_submission = copy.deepcopy(sample_submission)
for end_id, end_date in enumerate(end_data_list):
    total_loss =0.
    for e, (code, store) in enumerate(tqdm(zip(stock_list['종목코드'].values, storage.values()),total=len(range(370)))):
            if e in nyear:
                start_date = '20210104'
            else:
                start_date = '20160104'
            start_weekday = pd.to_datetime(start_date).weekday()
            max_weeknum = pd.to_datetime(end_date).strftime('%V')
            Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
            data = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
            data = pd.merge(Business_days, data, how='outer')
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
            data['weeknum'] = data['weeknum'].map(lambda x: int(x))
            data['week'] = data.weeknum

            Data=Make_Data(data)
            week = Data['week']
            data = Data[['Date', 'Close']]
            data.columns = ['ds', 'y']
            if end_id == 0:
                train = data[:-5]
                week = week[:-5]
                test_data = data.iloc[-5:, 1]
                plast_data = data.iloc[-7, 1]
                last_data = data.iloc[-6, 1]


            else:
                train = data
                test_data = data.iloc[-5:, 1]
                plast_data = data.iloc[-2, 1]
                last_data = data.iloc[-1, 1]

            pt = np.abs(last_data - plast_data) / last_data
            vol = np.abs(test_data.mean() - last_data) / last_data
            if vol < 0.28:
                if store == 1:
                    model = ARIMA(train['y'], order=(1, 1, 0), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터

                elif store == 2:
                    model = ARIMA(train['y'], order=(0, 1, 1), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터

                elif store == 3:
                    ex = week.iloc[-1]
                    ex = np.array([ex+1,ex+1,ex+1,ex+1,ex+1])
                    model = ARIMA(train['y'], order=(1, 1, 0),exog=week.iloc[:])
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5,exog=ex)  # 마지막 5일의 예측 데이터

                elif store == 4:
                    model = ARIMA(train['y'], order=(1, 0, 0), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터

                elif store== 5:
                    def moving_average(x, w):
                        return np.convolve(x, np.ones(w), 'valid') / w
                    model = ARIMA(train['y'], order=(1, 1, 0),)
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=10)  # 마지막 5일의 예측 데이터
                    forecast_data = moving_average(np.array(forecast_data).reshape(-1), 6)

                elif store == 6:
                    model = ARIMA(np.log1p(train['y']), order=(1, 1, 0), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
                    forecast_data = np.exp(forecast_data) + 1

                elif store == 7:
                    model = ARIMA(train['y'], order=(1, 2, 0))
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터

                elif store == 8:
                    start = Data.iloc[-1]
                    start = start['Close']
                    x = Data.iloc[:].diff().dropna()
                    x_train = x[['Close', 'High', 'Low', 'Open', 'Change']]
                    mod = sm.tsa.VAR(x_train)
                    res = mod.fit()
                    lag_order = res.k_ar
                    x_test = x_train.values[-lag_order:]
                    forecast_data = res.forecast(steps=5, y=x_test)
                    forecast_data = start + forecast_data[:, 0]

                elif store ==9:
                    model = ARIMA(train['y'].rolling(window=1).mean(), order=(1, 1, 0), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터

                elif store == 10:
                    train = train.ffill()
                    train = train.bfill()
                    model = ARIMA(train['y'], order=(1, 1, 0), )
                    model_fit = model.fit(method='yule_walker')
                    forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터

                elif store == 11:
                    model = ARIMA(train['y'], order=(1, 1, 0), trend=[0, 1])
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터


                else:
                    model = ARIMA(train['y'], order=(1, 1, 0))
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터

            else:
                start_date = '20210104'
                start_weekday = pd.to_datetime(start_date).weekday()
                max_weeknum = pd.to_datetime(end_date).strftime('%V')
                Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
                data = fdr.DataReader(code, start=start_date, end=end_date).reset_index()
                data = pd.merge(Business_days, data, how='outer')
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
                data['weeknum'] = data['weeknum'].map(lambda x: int(x))
                data['week'] = data.weeknum

                data = Make_Data(data)
                data = data.ffill()
                data = data.bfill()
                data = data[['Date', 'Close']]
                data.columns = ['ds', 'y']
                if end_id == 0:
                    train = data[:-5]
                else:
                    train = data[:]
                dtrain = train.copy()
                dtrain.iloc[1:, 1] = np.diff(dtrain.iloc[:, 1].values)
                dtrain.iloc[0, 1] = np.array(0)
                model = NeuralProphet(
                    n_forecasts=1,
                    n_lags=3,
                    changepoints_range=1,
                    n_changepoints=0,
                    num_hidden_layers=1,
                    d_hidden=1,
                )

                model.fit(dtrain,
                          freq='D',
                          progress_print=False, progress_bar=False, plot_live_loss=False)

                forecast_nn = model.predict(dtrain)
                forecast_yhat = forecast_nn[['yhat1']]
                forecast_yhat = np.array(forecast_yhat.iloc[-1], dtype=np.float).reshape(-1)
                base = np.array([data.iloc[-1, 1]])
                forecast_yhat = base + forecast_yhat
                forecast_yhat = np.tile(forecast_yhat, 5)
                forecast_data = forecast_yhat

            if end_id == 0:
                sample_submission[code][:5] = np.array(forecast_data).tolist()
            else:
                sample_submission[code][5:] = np.array(forecast_data).tolist()

    columns = list(sample_submission.columns[1:])
    columns = ['Day'] + [str(x).zfill(6) for x in columns]
    sample_submission.columns = columns
    sample_submission.to_csv('./all_Arima.csv',index=False)