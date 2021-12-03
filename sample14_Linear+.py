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
from xgboost import plot_importance
from matplotlib import pyplot
def nmae(true,pred):
    score = np.mean(np.abs(true-pred) / true) * 100
    return score
from sklearn.metrics import mean_absolute_error
path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path,list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))

# start_date = '20210104'
start_date = '20150104'
end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']
# end_data_list = ['20211029','20211105','20211112']
import catboost as cat

def Make_Data(data):

    keep = []
    for e, i in enumerate(data):
        if (i['weekday']==0) and (i['weeknum']==1):
            keep.append(e)
        if e == len(data)-1:
            keep.append(e)

    date = []
    for k in keep:
        date.append(data.loc[k,'Date'])

    df = []
    for d in range(len(date)-1):
        df.append(data.query("Date >= '{}' and Date < '{}'".format(date[d],date[d+1])))

    ord = []
    for d in df:
        ord.append(d['weeknum'].iloc[-1])

    ord1 = []
    base = 0
    for d in ord:
        base += d
        ord1.append(base)

    frame = pd.DataFrame()
    for d, b in zip(df,ord1):
        d['weeknum'] = d['weeknum'].map(lambda x: x + b)
        pd.concat([frame,d],ignore_index=True)

    return frame

for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    model = LinearRegression()
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

        data['weeknum'] = data['weeknum'].map(lambda x: int(x))
        data['week'] = data.weeknum
        print(data)


        data2016 = data.query("Date >= '2016-01-04' and Date <= '2016-12-30'")
        data2017 = data.query("Date >= '2017-01-02' and Date <= '2017-12-29'")
        data2018 = data.query("Date >= '2018-01-01' and Date <= '2018-12-28'")
        data2019 = data.query("Date >= '2019-01-07' and Date <= '2019-12-27'")
        data2020 = data.query("Date >= '2019-12-30' and Date <= '2021-01-01'")
        data2021 = data.query("Date >= '2021-01-04'")

        wd2016 = data2016['weeknum'].iloc[-1]
        wd2017 = data2017['weeknum'].iloc[-1]
        wd2018 = data2018['weeknum'].iloc[-1]
        wd2019 = data2019['weeknum'].iloc[-1]
        wd2020 = data2020['weeknum'].iloc[-1]
        wd2021 = data2021['weeknum'].iloc[-1]

        data2017['weeknum'] = data2017['weeknum'].map(lambda x: x + wd2016)
        data2018['weeknum'] = data2018['weeknum'].map(lambda x: x + wd2016 + wd2017)
        data2019['weeknum'] = data2019['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018)
        data2020['weeknum'] = data2020['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018+wd2019)
        data2021['weeknum'] = data2021['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018+wd2019+wd2020)

        data = pd.concat([data2021],axis=0)
        # data = pd.concat([data2021], axis=0)
        # 5.4409138029826
        #
        # 5.749088442530942
        # 4.644561298856667
        data = pd.pivot_table(data=data, values=['Close','High','Low',
                                                 'Open','Volume','Change','week'],
                              columns='weekday', index='weeknum')

        data = data[['Close','High','Low','Open','week']]
        data = data.dropna(axis=0)
        test_submission.loc[:, code] = pd.concat([data.iloc[-1,:5], data.iloc[-1,:5]], ignore_index=True)


        x_data = data.iloc[0:-2]
        y_data = data.iloc[1:-1,:5]
        x_data = np.array(x_data).reshape(-1,25)
        y_data = np.array(y_data)
        x_public = data.iloc[-2].to_numpy().reshape(1, 25)  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.
        # obj=nmae, feval=nmae, verbosity=0, silent=0,verbose_eval=None

        y_0 = y_data[:, 0]
        y_1 = y_data[:, 1]
        y_2 = y_data[:, 2]
        y_3 = y_data[:, 3]
        y_4 = y_data[:, 4]

        y_values = [y_0, y_1, y_2, y_3, y_4]

        predictions = []
        for y_value in y_values:
            model.fit(x_data,y_value)
            # plot_importance(model)
            # pyplot.show()
            pred = model.predict(x_public)
            predictions.append(pred[0])
        # print(predictions)
        # print(sample_submission.loc[:, code])
        sample_submission.loc[:, code] = predictions * 2
    # sample_submission.isna().sum().sum()

    columns = list(sample_submission.columns[1:])

    columns = ['Day'] + [str(x).zfill(6) for x in columns]

    sample_submission.columns = columns
    # print(test_submission.iloc[:, 1:].values)
    # print(sample_submission.iloc[:, 1:].values)
    # print(sample_submission.isna().sum().sum())
    # print(mean_absolute_error(test_submission.iloc[:, 1:].values, sample_submission.iloc[:, 1:].values))
    print(nmae(test_submission.iloc[:, 1:].values, sample_submission.iloc[:, 1:].values))


#2016
'''
100%|██████████| 370/370 [00:53<00:00,  6.93it/s]
4.484522231288025
100%|██████████| 370/370 [00:52<00:00,  7.05it/s]
4.395337841803168
100%|██████████| 370/370 [00:53<00:00,  6.93it/s]
3.4597359690433303
100%|██████████| 370/370 [00:52<00:00,  7.01it/s]
3.435327782167201
100%|██████████| 370/370 [00:53<00:00,  6.91it/s]
3.247523767890027
100%|██████████| 370/370 [00:52<00:00,  7.03it/s]
3.8797456510565196
100%|██████████| 370/370 [00:52<00:00,  7.00it/s]
4.244058300587263
'''
#2017
'''
4.478611679907788
100%|██████████| 370/370 [00:53<00:00,  6.96it/s]
4.458293821294984
100%|██████████| 370/370 [00:52<00:00,  7.01it/s]
3.556735015618672
100%|██████████| 370/370 [00:52<00:00,  6.98it/s]
3.491679181447339
100%|██████████| 370/370 [00:52<00:00,  7.09it/s]
3.2914013704420015
100%|██████████| 370/370 [00:52<00:00,  7.04it/s]
3.970761707262571
100%|██████████| 370/370 [00:52<00:00,  7.07it/s]
4.306785769403118
'''
#2018
'''
100%|██████████| 370/370 [00:52<00:00,  7.04it/s]
4.543061947384701
100%|██████████| 370/370 [00:52<00:00,  7.08it/s]
4.546512517327516
100%|██████████| 370/370 [00:52<00:00,  6.99it/s]
3.662952785861426
100%|██████████| 370/370 [00:52<00:00,  7.08it/s]
3.5940080947776556
100%|██████████| 370/370 [00:52<00:00,  7.05it/s]
3.3370100098758244
100%|██████████| 370/370 [00:52<00:00,  7.08it/s]
4.096981646133345
100%|██████████| 370/370 [00:52<00:00,  6.98it/s]
4.33957734469519
'''