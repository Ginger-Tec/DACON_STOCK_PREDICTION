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

# start_date = '20210104'
start_date = '20160104'
# end_data_list = ['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']
end_data_list =['20211105']
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
            x = data['Close']
            test_submission.loc[:, code] = pd.concat([x[-5:], data.iloc[-5:]], ignore_index=True)
            train = x[:-5]
            train = np.array(train)

            train = train.reshape(-1)
            model = ARIMA(train, order=(1, 1, 0))
            model_fit = model.fit()
            forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
            sample_submission.loc[:, code] = np.array(forecast_data).tolist() * 2

        # print(nmae(x[-5:], forecast_data))
        # data = pd.pivot_table(data=data, values=['Close','High','Low','Open','Volume','Change','week'], columns='weekday', index='weeknum')
        #
        # data = data[['Close','High','Low','Open','week']]
        # data = data.dropna(axis=0)
        # test_submission.loc[:, code] = pd.concat([data.iloc[-1,:5], data.iloc[-1,:5]], ignore_index=True)
        #
        #
        # x_data = data.iloc[0:-2]
        # y_data = data.iloc[1:-1,:5]
        # x_data = np.array(x_data).reshape(-1,25)
        # y_data = np.array(y_data)
        # x_public = data.iloc[-2].to_numpy().reshape(1, 25)  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.
        # # obj=nmae, feval=nmae, verbosity=0, silent=0,verbose_eval=None
        #
        # for col in list(x_data.columns):
        #     print(col)
        #     model = ARIMA(x_data[col], order=(2, 0, 0))
        #     model_fit = model.fit(trend='c', full_output=True, disp=True)
    #
    #     y_0 = y_data[:, 0]
    #     y_1 = y_data[:, 1]
    #     y_2 = y_data[:, 2]
    #     y_3 = y_data[:, 3]
    #     y_4 = y_data[:, 4]
    #
    #     y_values = [y_0, y_1, y_2, y_3, y_4]
    #
    #     predictions = []
    #     for y_value in y_values:
    #         model.fit(x_data,y_value)
    #         # plot_importance(model)
    #         # pyplot.show()
    #         pred = model.predict(x_public)
    #         predictions.append(pred[0])
    #     # print(predictions)
    #     # print(sample_submission.loc[:, code])
    #     sample_submission.loc[:, code] = predictions * 2
    # sample_submission.isna().sum().sum()

    columns = list(sample_submission.columns[1:])

    columns = ['Day'] + [str(x).zfill(6) for x in columns]

    sample_submission.columns = columns
    sample_submission.to_csv('./just_Arima.csv',index=False)
    # # print(test_submission.iloc[:, 1:].values)
    # # print(sample_submission.iloc[:, 1:].values)
    # # print(sample_submission.isna().sum().sum())
    # # print(mean_absolute_error(test_submission.iloc[:, 1:].values, sample_submission.iloc[:, 1:].values))
    # print(nmae(test_submission.iloc[:, 1:].values, sample_submission.iloc[:, 1:].values))
#Linear
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
#ARIMA(2,0,0)
'''
100%|██████████| 370/370 [02:21<00:00,  2.62it/s]
4.1041756395559705
100%|██████████| 370/370 [02:15<00:00,  2.74it/s]
3.3516765182157555
100%|██████████| 370/370 [02:13<00:00,  2.76it/s]
2.478232488165112
100%|██████████| 370/370 [02:17<00:00,  2.69it/s]
2.6768755707934506
100%|██████████| 370/370 [02:15<00:00,  2.73it/s]
  0%|          | 0/370 [00:00<?, ?it/s]2.820244722834014
100%|██████████| 370/370 [02:18<00:00,  2.66it/s]
3.4840219939794226
100%|██████████| 370/370 [02:17<00:00,  2.70it/s]
3.6346867466849355
'''
#ARIMA(2,1,0)
'''
100%|██████████| 370/370 [01:25<00:00,  4.30it/s]
4.164241124911417
100%|██████████| 370/370 [01:26<00:00,  4.27it/s]
3.4029378879934353
100%|██████████| 370/370 [01:25<00:00,  4.34it/s]
2.463427756296171
100%|██████████| 370/370 [01:25<00:00,  4.34it/s]
2.6435431184794447
100%|██████████| 370/370 [01:25<00:00,  4.33it/s]
2.8081603078693513
100%|██████████| 370/370 [01:25<00:00,  4.34it/s]
2.976520820788242
100%|██████████| 370/370 [01:25<00:00,  4.33it/s]
3.6016190073680403
'''
#(1,1,0)
'''
100%|██████████| 370/370 [01:13<00:00,  5.07it/s]
4.17065217880493
100%|██████████| 370/370 [01:12<00:00,  5.11it/s]
3.3828219213567867
100%|██████████| 370/370 [01:12<00:00,  5.10it/s]
2.440931666591573
100%|██████████| 370/370 [01:12<00:00,  5.11it/s]
2.645908741845233
100%|██████████| 370/370 [01:12<00:00,  5.08it/s]
2.80874560469795
100%|██████████| 370/370 [01:12<00:00,  5.08it/s]
2.9709199374510593
100%|██████████| 370/370 [01:12<00:00,  5.08it/s]
3.601202447801195
'''
#(0,1,1)
'''
100%|██████████| 370/370 [01:20<00:00,  4.59it/s]
4.173708467213374
100%|██████████| 370/370 [01:18<00:00,  4.73it/s]
3.4010342565437783
100%|██████████| 370/370 [01:18<00:00,  4.72it/s]
2.440892045453292
100%|██████████| 370/370 [01:18<00:00,  4.74it/s]
2.645007741510734
100%|██████████| 370/370 [01:17<00:00,  4.76it/s]
  0%|          | 0/370 [00:00<?, ?it/s]2.8076470503152646
100%|██████████| 370/370 [01:17<00:00,  4.75it/s]
2.967761173851035
100%|██████████| 370/370 [01:17<00:00,  4.76it/s]
3.604963054766575
'''
#(1,1,1)
'''
100%|██████████| 370/370 [02:09<00:00,  2.85it/s]
4.202750002269898
100%|██████████| 370/370 [02:09<00:00,  2.85it/s]
3.422259646685533
100%|██████████| 370/370 [02:08<00:00,  2.89it/s]
2.4710760574462154
100%|██████████| 370/370 [02:08<00:00,  2.88it/s]
2.65708769237639
100%|██████████| 370/370 [02:09<00:00,  2.87it/s]
2.7946348571729964
100%|██████████| 370/370 [02:09<00:00,  2.86it/s]
2.970293140062823
100%|██████████| 370/370 [02:11<00:00,  2.82it/s]
3.60282364803839
'''
#log(0,1,1)
'''
100%|██████████| 370/370 [01:52<00:00,  3.29it/s]
4.172432074015483
100%|██████████| 370/370 [01:52<00:00,  3.30it/s]
3.394542551852294
100%|██████████| 370/370 [01:51<00:00,  3.32it/s]
2.432703462777204
100%|██████████| 370/370 [01:51<00:00,  3.31it/s]
2.6450805376727855
100%|██████████| 370/370 [01:51<00:00,  3.31it/s]
2.8175493084658236
100%|██████████| 370/370 [01:53<00:00,  3.27it/s]
2.9670056454312923
100%|██████████| 370/370 [01:53<00:00,  3.26it/s]
3.6067276705940365
'''
#log(1,1,0)
'''
100%|██████████| 370/370 [01:38<00:00,  3.76it/s]
4.166534707209618
100%|██████████| 370/370 [01:38<00:00,  3.77it/s]
3.3868219232936263
100%|██████████| 370/370 [01:36<00:00,  3.84it/s]
2.4302503972061666
100%|██████████| 370/370 [01:35<00:00,  3.86it/s]
2.648063671736589
100%|██████████| 370/370 [01:37<00:00,  3.78it/s]
  0%|          | 0/370 [00:00<?, ?it/s]2.8202525934663307
100%|██████████| 370/370 [01:37<00:00,  3.79it/s]
2.969395665314937
100%|██████████| 370/370 [01:38<00:00,  3.77it/s]
3.6039897381338304
'''