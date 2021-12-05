import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
import copy
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import warnings
from neuralprophet import NeuralProphet
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import datetime
from collections import defaultdict,Counter
import pickle
device = torch.device('cuda')
from matplotlib import pyplot
def nmae(true, pred):
    score = np.mean(np.abs(true - pred) / true) * 100
    return score

path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'
stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))


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

storage = defaultdict(list)

end_stock_date = '20211029' #이번주 주식 끝나는 날 입력
year = end_stock_date[:4]
month = end_stock_date[4:6]
day = end_stock_date[6:]
end_datetime = datetime.date(int(year), int(month), int(day))

end_data_list = []
for e in range(7):
    diff_days = datetime.timedelta(days=7)
    end_datetime = end_datetime - diff_days
    ed = str(end_datetime)
    ed = ed.split('-')
    ed = ''.join(ed)
    end_data_list.append(ed)
end_data_list.reverse()


year = end_stock_date[:4]
month = end_stock_date[4:6]
day = end_stock_date[6:]
end_datetime = datetime.date(int(year), int(month), int(day))

eval_stock_date = []
diff_days = datetime.timedelta(days=7)
end_datetime = end_datetime + diff_days
ed = str(end_datetime)
ed = ed.split('-')
ed = ''.join(ed)#실제 평가받는 주의 마지막 날(금요일)
eval_stock_date.append(ed)

tse=[]
for end_date in end_data_list:
    total_loss =0.
    se = []
    for e, code in enumerate(tqdm(stock_list['종목코드'].values)):
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
            data=Make_Data(data)

            x = data[['Close', 'week']]
            train = x
            model = ARIMA(train.iloc[:, 0], order=(1, 1, 0))
            model_fit = model.fit()
            forecast_data = model_fit.forecast(steps=5,)  # 마지막 5일의 예측 데이터
            tmp = nmae(x.iloc[-5:, 0], np.array(forecast_data).reshape(-1))

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

            x = data[['Close', 'week']]
            train = x
            model = ARIMA(train.iloc[:, 0], order=(1, 1, 0))
            model_fit = model.fit()
            forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
            tmp2 = nmae(x.iloc[-5:, 0], np.array(forecast_data).reshape(-1))
            if tmp2 < tmp:
                se.append(e)
    tse.append(se)
def extract(tse):
    a1 = tse[0]
    a2 = tse[1]
    a3 = tse[2]
    a4 = tse[3]
    a5 = tse[4]
    a6 = tse[5]
    a7 = tse[6]
    intersection = list(set(a1) & set(a2))
    intersection = list(set(intersection) & set(a3))
    intersection = list(set(intersection) & set(a4))
    intersection = list(set(intersection) & set(a5))
    intersection = list(set(intersection) & set(a6))
    intersection = list(set(intersection) & set(a7))
    return intersection
nyear = extract(tse)
ts2, ts3, ts4, ts5, ts6, ts7, ts8, ts9, ts10, ts11 = [], [], [], [], [], [], [], [], [], []
for end_id, end_date in enumerate(end_data_list):
    total_loss =0.
    s2, s3, s4, s5, s6, s7, s8, s9, s10, s11 = [], [], [], [], [], [], [], [], [], []
    for e, code in enumerate(tqdm(stock_list['종목코드'].values)):
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
            train = data[:-5]
            week = week[:-5]
            test_data = data.iloc[-5:, 1]
            tmp_list = []

            model = ARIMA(train['y'], order=(1, 1, 0), )
            model_fit = model.fit()
            forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
            base_loss = nmae(test_data, np.array(forecast_data).reshape(-1))

            for store in range(2,12):
                if store == 2:
                    model = ARIMA(train['y'], order=(0, 1, 1), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s2.append(e)

                if store == 3:
                    ex = week.iloc[-1]
                    ex = np.array([ex+1,ex+1,ex+1,ex+1,ex+1])
                    model = ARIMA(train['y'], order=(1, 1, 0),exog=week.iloc[:])
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5,exog=ex)  # 마지막 5일의 예측 데이터
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s3.append(e)

                if store == 4:
                    model = ARIMA(train['y'], order=(1, 0, 0), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s4.append(e)

                if store== 5:
                    def moving_average(x, w):
                        return np.convolve(x, np.ones(w), 'valid') / w
                    model = ARIMA(train['y'], order=(1, 1, 0),)
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=10)  # 마지막 5일의 예측 데이터
                    forecast_data = moving_average(np.array(forecast_data).reshape(-1), 6)
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s5.append(e)

                if store == 6:
                    model = ARIMA(np.log1p(train['y']), order=(1, 1, 0), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
                    forecast_data = np.exp(forecast_data) + 1
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s6.append(e)

                if store == 7:
                    model = ARIMA(train['y'], order=(1, 2, 0))
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s7.append(e)

                if store == 8:
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
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s8.append(e)
                if store ==9:
                    model = ARIMA(train['y'].rolling(window=1).mean(), order=(1, 1, 0), )
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s9.append(e)
                if store ==  10:
                    train = train.ffill()
                    train = train.bfill()
                    model = ARIMA(train['y'], order=(1, 1, 0), )
                    model_fit = model.fit(method='yule_walker')
                    forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s10.append(e)
                if store == 11:
                    model = ARIMA(train['y'], order=(1, 1, 0), trend=[0, 1])
                    model_fit = model.fit()
                    forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터
                    tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
                    tmp_list.append(tmp)
                    if tmp < base_loss:
                        s11.append(e)

            tmp_index = np.argmin(tmp_list)
            storage[e].append(int(tmp_index))
    ts2.append(s2); ts3.append(s3); ts4.append(s4); ts5.append(s5); ts6.append(s6)
    ts7.append(s7); ts8.append(s8); ts9.append(s9); ts10.append(s10); ts11.append(s11)
e2 = extract(ts2); e3 = extract(ts3); e4 = extract(ts4); e5 = extract(ts5); e6 = extract(ts6)
e7 = extract(ts7); e8 = extract(ts8); e9 = extract(ts9); e10 = extract(ts10); e11 = extract(ts11)

for id, e in enumerate([e2,e3,e4,e5,e6,e7,e8,e9,e10,e11]):
    with open(f'./e{id+2}','wb') as f:
        pickle.dump(e,f)

with open('./storage.pkl','wb') as f:
    pickle.dump(storage,f)

with open(f'./e2','rb') as f:
    e2 = pickle.load(f)
with open(f'./e3', 'rb') as f:
    e3 = pickle.load(f)
with open(f'./e4', 'rb') as f:
    e4 = pickle.load(f)
with open(f'./e5', 'rb') as f:
    e5 = pickle.load(f)
with open(f'./e6', 'rb') as f:
    e6 = pickle.load(f)
with open(f'./e7', 'rb') as f:
    e7 = pickle.load(f)
with open(f'./e8', 'rb') as f:
    e8 = pickle.load(f)
with open(f'./e9', 'rb') as f:
    e9 = pickle.load(f)
with open(f'./e10', 'rb') as f:
    e10 = pickle.load(f)
with open(f'./e11', 'rb') as f:
    e11 = pickle.load(f)
with open('./storage.pkl','rb') as f:
    rank = pickle.load(f)

storage = {}
method = {}
for i in range(370):
    storage[i] = 1

for id, te in enumerate([e2, e3, e4, e5, e6, e7, e8, e9, e10, e11]):
    for e in te:
        if storage[e] == 1:
            storage[e] = id+2
            method[e] = id+2
        else:
            tmp = Counter(rank[id])
            value = tmp[id+2]
            com = tmp[method[e]]
            if value > com:
                storage[e] = id+2
                method[e] = id+2

for end_id, end_date in enumerate(eval_stock_date):
    total_loss = 0.
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
            train = data[:-5]
            week = week[:-5]
            test_data = data.iloc[-5:, 1]
            plast_data = data.iloc[-7, 1]
            last_data = data.iloc[-6, 1]
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

            tmp = nmae(test_data, np.array(forecast_data).reshape(-1))
            total_loss += tmp

    #최종점수 평가
    print(total_loss/370)


    #         if end_id == 0:
    #             sample_submission[code][:5] = np.array(forecast_data).tolist()
    #         else:
    #             sample_submission[code][5:] = np.array(forecast_data).tolist()
    #
    # columns = list(sample_submission.columns[1:])
    # columns = ['Day'] + [str(x).zfill(6) for x in columns]
    # sample_submission.columns = columns
    # sample_submission.to_csv('./all_Arima.csv',index=False)

'''
['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']

#1105

#1112
3.599326758137693
'''