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
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

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
import arch
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
# end_data_list = ['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']
# end_data_list = ['20211119']
# end_data_list = ['20211126']
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

tse=[]
for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    total_loss =0.
    se = []
    for e, code in enumerate(tqdm(stock_list['종목코드'].values)):
        # if e==113:
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

            x = data[['Close', 'week']]
            # x = data[['Close','week']]

            train = x[:-5]

            # train = np.array(train)
            #AR-exog
            # ex = train.iloc[-1,1]
            # ex = np.array([ex+1,ex+1,ex+1,ex+1,ex+1])
            # model = ARIMA(train.iloc[:,0], order=(1, 1, 0),exog=train.iloc[:,1])
            # model_fit = model.fit()
            # forecast_data = model_fit.forecast(steps=5,exog=ex)  # 마지막 5일의 예측 데이터
            # tmp2 = nmae(x.iloc[-5:,0], np.array(forecast_data).reshape(-1))
            # print(tmp)

            # AR
            # model = ARIMA(train.iloc[:, 0].rolling(window=1).mean(), order=(1, 1, 0),)
            # model_fit = model.fit()
            # forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
            # tmp2 = nmae(x.iloc[-5:, 0], np.array(forecast_data).reshape(-1))


            # def moving_average(x, w):
            #     return np.convolve(x, np.ones(w), 'valid') / w
            #
            # # AR
            # model = ARIMA(np.log1p(train.iloc[:, 0]), order=(1, 1, 0), )
            # model_fit = model.fit()
            # forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
            # tmp2 = nmae(x.iloc[-5:, 0], np.exp(np.array(forecast_data).reshape(-1)) + 1)



            # AR 2
            model = ARIMA(train.iloc[:, 0], order=(1, 1, 0))
            model_fit = model.fit()
            forecast_data = model_fit.forecast(steps=5,)  # 마지막 5일의 예측 데이터
            tmp = nmae(x.iloc[-5:, 0], np.array(forecast_data).reshape(-1))


            #AR - c
            # model = ARIMA(train.iloc[:, 0], order=(1, 1, 0),trend=[0,1])
            # model_fit = model.fit()
            # forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
            # tmp2 = nmae(x.iloc[-5:, 0], np.array(forecast_data).reshape(-1))



            #RandomForestRegressor
            model = RandomForestRegressor()
            data = pd.pivot_table(data=data, values='Close', columns='weekday', index='weeknum')
            data = data.ffill()
            data = data.bfill()
            X = data.iloc[0:-2].to_numpy()
            y = data.iloc[1:-1].to_numpy()
            y_0 = y[:, 0]
            y_1 = y[:, 1]
            y_2 = y[:, 2]
            y_3 = y[:, 3]
            y_4 = y[:, 4]

            y_values = [y_0, y_1, y_2, y_3, y_4]
            x_public = data.iloc[-2].to_numpy()

            predictions = []
            for y_value in y_values:
                model.fit(X, y_value)
                prediction = model.predict(np.expand_dims(x_public, 0))
                predictions.append(prediction[0])
            tmp2 = nmae(x.iloc[-5:, 0], np.array(predictions).reshape(-1))
            # # AR
            # model = ARIMA(train.iloc[:, 0], order=(1, 1, 0),)
            # model_fit = model.fit()
            # forecast_data = model_fit.forecast(steps=10)  # 마지막 5일의 예측 데이터
            # tmp2 = nmae(x.iloc[-5:, 0], moving_average(np.array(forecast_data).reshape(-1),6))
            #VAR
            # start = data.iloc[-6]
            # start = start['Close']
            #
            # x = data.iloc[:-5].diff().dropna()
            # public = data['Close']
            # public = public[-5:].values
            # x_train = x[['Close', 'High', 'Low', 'Open', 'Change']]
            # mod = sm.tsa.VAR(x_train)
            # # mod = sm.tsa.VARMAX(x[['Close','Open', 'High','Low']], order=(1, 0), trend='c')
            # # mod = sm.tsa.VARMAX(x[['Close','Open', 'High','Low']], order=(0, 1), error_cov_type='diagonal')
            # res = mod.fit()
            # lag_order = res.k_ar
            # x_test = x_train.values[-lag_order:]
            # forecast_data = res.forecast(steps=5, y=x_test)
            # forecast_data = start + forecast_data[:, 0]
            # tmp2 = nmae(public, forecast_data)

            # AR
            # model = ARIMA(train.iloc[:, 0], order=(1, 2, 0), )
            # model_fit = model.fit()
            # forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
            # tmp2 = nmae(x.iloc[-5:, 0], np.array(forecast_data).reshape(-1))

            #MA
            # model = ARIMA(train.iloc[:, 0], order=(0, 1, 1), )
            # model_fit = model.fit()
            # forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
            # tmp2 = nmae(x.iloc[-5:, 0], np.array(forecast_data).reshape(-1))

            # print(tmp, tmp2)
#
#             # print(tmp,tmp2)
# #             if 2> tmp:
# #                 se.append(e)
#
            if tmp2 < tmp:
                se.append(e)
#
            # total_loss += tmp
    tse.append(se)
    # print(total_loss/370)#             total_loss += tmp2


a1 = tse[0]
a2 = tse[1]
a3 = tse[2]
a4 = tse[3]
a5 = tse[4]
a6 = tse[5]
a7 = tse[6]
intersection = list(set(a1) & set(a2))
print(intersection)
intersection = list(set(intersection) & set(a3))
print(intersection)
intersection = list(set(intersection) & set(a4))
print(intersection)
intersection = list(set(intersection) & set(a5))
print(intersection)
intersection = list(set(intersection) & set(a6))
print(intersection)

#250 예측하기 어려움

#[113, 258, 212, 206] MA
#[46, 31] exog
#[288, 257, 133, 229, 295, 123] 차분x
#2이하 [68, 134, 168, 72, 170, 360, 76, 173, 142, 334, 50, 178, 149, 125, 23, 184, 317]
#4이상 [250, 255]
#210 mv
#[325, 6, 234, 333, 240, 345] 후 mv
#[322, 260, 325, 46] log