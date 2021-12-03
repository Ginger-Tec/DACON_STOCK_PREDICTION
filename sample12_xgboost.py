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
start_date = '20160104'
end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']
# end_data_list = ['20211029','20211105','20211112']
import catboost as cat

for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    # model = LinearRegression()
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

        data = pd.concat([data2016,data2017,data2018,data2019,data2020,data2021],axis=0)
        # data = pd.concat([data2021], axis=0)
        # 5.4409138029826
        #
        # 5.749088442530942
        # 4.644561298856667
        data = pd.pivot_table(data=data, values=['Close','High','Low','Open','Volume','Change','week'], columns='weekday', index='weeknum')

        data = data[['Close']]
        data = data.dropna(axis=0)
        test_submission.loc[:, code] = pd.concat([data.iloc[-1,:5], data.iloc[-1,:5]], ignore_index=True)

        x = data.iloc[0:-1].to_numpy()  # 2021년 1월 04일 ~ 2021년 10월 22일까지의 데이터로

        # x_data = []
        # y_data = []
        #
        # for r in range(1000000):
        #     if r + 1 == len(x):
        #         break
        #     x_data.append(x[r:r + 1])
        #     y_data.append(x[r + 1, :5])
        x_data = data.iloc[0:-3]
        x_data2 = data.iloc[1:-2]
        x_data = pd.concat([x_data,x_data2],axis=1,ignore_index=True)
        y_data = data.iloc[1:-1,:5]
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        x_public_pre = data.iloc[-3].to_numpy().reshape(1,-1)
        x_public = data.iloc[-2].to_numpy().reshape(1, -1)  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.
        x_public = np.concatenate([x_public_pre,x_public],axis=1)
        # obj=nmae, feval=nmae, verbosity=0, silent=0,verbose_eval=None
        model = xgb.XGBRegressor()
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

        y1 = y_train[:,0]
        y2 = y_train[:,1]
        y3 = y_train[:,2]
        y4 = y_train[:,3]
        y5 = y_train[:,4]

        y1t = y_test[:, 0]
        y2t = y_test[:, 1]
        y3t = y_test[:, 2]
        y4t = y_test[:, 3]
        y5t = y_test[:, 4]

        predictions = []
        for t,tx in zip([y1,y2,y3,y4,y5],[y1t,y2t,y3t,y4t,y5t]):
            model.fit(x_train, t, eval_set=[(x_test, tx)], early_stopping_rounds=50, verbose=False)
            # plot_importance(model)
            # pyplot.show()
            pred = model.predict(x_public)
            predictions.append(pred[0])
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

# 0.02741967519599727
# 4631.059395323057