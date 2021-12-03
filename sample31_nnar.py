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
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import random
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

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def nmae(true, pred):
    score = np.mean(np.abs(true - pred) / true) * 100
    return score
def NMAE(true,pred):
    score = torch.mean(torch.abs(true-pred) / true) * 100
    return score

from sklearn.metrics import mean_absolute_error

path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))

# start_date = '20210104'
# start_date = '20160104'
# end_data_list = ['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']
# end_data_list = ['20211119']
# end_data_list = ['20211029','20211105','20211112']
# end_data_list = ['20211112']
end_data_list = ['20211105']
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

seed_everything(seed=103)

for end_date in end_data_list:
    # print('businees', Business_days)

    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    total_loss =0.
    se = []

    a1=0.
    a2=0.
    a3=0.


    for e, code in enumerate(tqdm(stock_list['종목코드'].values)):
        # if 100<e<110:
            start_date = '20160104'
            start_weekday = pd.to_datetime(start_date).weekday()
            max_weeknum = pd.to_datetime(end_date).strftime('%V')
            Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

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

            data = Make_Data(data)
            data = data.ffill()
            data = data.bfill()
            data = data[['Date', 'Close']]
            data.columns = ['ds', 'y']
            data = data.ffill()
            data = data.bfill()
            train = data[:-5]
            # print(train)
            test_data = data.iloc[-10:-5, 1]
            plast_data = data.iloc[-7, 1]
            last_data = data.iloc[-6, 1]
            pt = np.abs(last_data-plast_data) / last_data
            # print(pt)
            vol = np.abs(test_data.mean() - last_data) / last_data
            # if vol < 0.14:
            #     continue

            model = ARIMA(train.iloc[:, 1], order=(1, 1, 0), )
            model_fit = model.fit()
            forecast_data = model_fit.forecast(steps=5, )  # 마지막 5일의 예측 데이터



            start_date = '20210104'
            start_weekday = pd.to_datetime(start_date).weekday()
            max_weeknum = pd.to_datetime(end_date).strftime('%V')
            Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

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
            data = data[['Date', 'Close']]
            data.columns = ['ds','y']
            # print(data)

            train = data[:-5]
            dtrain = train.copy()
            dtrain.iloc[1:,1] = np.diff(dtrain.iloc[:,1].values)
            dtrain = dtrain.dropna(0)
            # dtrain = np.log1p(dtrain)
            model = NeuralProphet(
                n_forecasts=1,
                n_lags=1,
                changepoints_range=1,
                n_changepoints=0,
                num_hidden_layers=1,
                d_hidden=1,
            )
            model.fit(dtrain,
                      freq='D',
                      progress_print=False,progress_bar=False, plot_live_loss=False)
            forecast_nn = model.predict(dtrain)
            forecast_yhat = forecast_nn[['yhat1']]
            # forecast_yhat = forecast_nn[['yhat1','yhat2','yhat3','yhat4','yhat5']]
            forecast_yhat = np.array(forecast_yhat.iloc[-1],dtype=np.float).reshape(-1)
            # forecast_yhat = np.mean(np.array(forecast_yhat.iloc[-1], dtype=np.float).reshape(-1))
            base = np.array([data.iloc[-6,1]])
            # forecast_yhat = np.cumsum(forecast_yhat)
            forecast_yhat = base + forecast_yhat
            forecast_yhat = np.tile(forecast_yhat, 5)


            # check2 = forecast_yhat[0]
            # check1 = np.array(forecast_data)[0]
            # check = np.abs(np.array(data.iloc[-6,1])-check2) > (np.array(data.iloc[-6,1])-check1)
            tmp2 = nmae(data.iloc[-5:, 1], np.array(forecast_yhat).reshape(-1))
            tmp1 = nmae(data.iloc[-5:, 1], np.array(forecast_data).reshape(-1))
            # print(vol)
            # if check:
            #     a3 += tmp1
            #
            # else:
            #     a3 += tmp2

            # print(pt)
            # print(tmp1,tmp2)

            a1 += tmp1
            a2 += tmp2




    ####
         # total_loss += tmp
    #     tse.append(se)
    # print(total_loss/370)
    # tse.append(total_loss/370)
    #
    #         break
    #
    # break
# print(tse
'''
100%|██████████| 370/370 [00:00<00:00, 763.89it/s]
2.036983682641791 2.017611515035083
100%|██████████| 370/370 [00:00<00:00, 986.78it/s]
1.3902936387048734 1.379158572704943
100%|██████████| 370/370 [00:00<00:00, 1246.31it/s]
0.722591724073011 0.711545907322228
100%|██████████| 370/370 [00:00<00:00, 1029.56it/s]
2.6538842198154993 2.6361541218281936
100%|██████████| 370/370 [00:00<00:00, 1127.98it/s]
1.6740862596638078 1.670828319443301
100%|██████████| 370/370 [00:00<00:00, 1127.31it/s]
0.9602590518462749 0.9497805102619873
100%|██████████| 370/370 [00:00<00:00, 1076.37it/s]
0.4623877646920528 0.4659743766071111

1.4736314739500105
0.991249351894696
0.2958717630885452
3.377548964823923
1.4214540196511796
0.5841325846330631
1.1956585110493172
'''
# 3.1295575462279137
# 2.9691395102926674
# 3.1077771021841554
# 3.086858757296724
# 3.0670152483979405
# 3.0482894273987067
# 3.031444824690544
# 3.015848919988958
# 3.002017041274848
# 2.989027415707299
# 2.9780171706345335

#publick week
#2.9691395102926674
#3.074635004691494

#arima vs arnet
#3.6040254521079116
#3.584308093487665