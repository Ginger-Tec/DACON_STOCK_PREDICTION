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
from sklearn.feature_selection import SelectFromModel
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
# end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']
end_data_list = ['20211029']
import catboost as cat
sample_submission = pd.read_csv(os.path.join(path, sample_name))
test_submission = copy.deepcopy(sample_submission)
full_data = []
for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
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
        data2020['weeknum'] = data2020['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018 + wd2019)
        data2021['weeknum'] = data2021['weeknum'].map(lambda x: x + wd2016 + wd2017 + wd2018 + wd2019 + wd2020)

        data = pd.concat([data2016, data2017, data2018, data2019, data2020, data2021], axis=0)

        data = pd.pivot_table(data=data, values=['Close', 'High', 'Low', 'Open', 'Volume', 'Change', 'week'],
                              columns='weekday', index='weeknum')

        data = data[['Close', 'High', 'Low', 'Open', 'week']]
        data = data.dropna(axis=0)
        test_submission.loc[:, code] = pd.concat([data.iloc[-1, :5], data.iloc[-1, :5]], ignore_index=True)
        full_data.append(data)

full_data = pd.concat(full_data,axis=1)
# print(full_data.shape)

full_data = full_data.fillna(method='ffill')
# print(full_data.shape)
for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    # model = LinearRegression()
    num = 1
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

        data = data[['Close','High','Low','Open','week']]
        data = data.dropna(axis=0)
        # data = data.fillna('ffill')
        # test_submission.loc[:, code] = pd.concat([data.iloc[-1,:5], data.iloc[-1,:5]], ignore_index=True)
        if len(full_data) == len(data):
            tmp_data = full_data
        else:
            # print(full_data)
            # print(data)
            s_idx = set(full_data.index) - set(data.index)
            tmp_data = full_data.drop(index=list(s_idx))
        x_data = tmp_data.iloc[0:-2]
        y_data = data.iloc[1:-1,:5]
        x_data = np.array(x_data)
        x_public = tmp_data.iloc[-2].to_numpy().reshape(1, -1)  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.

        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=25)
        # print(x_train.shape)
        # print(y_train.shape)
        y1 = y_train.iloc[:,0]
        y2 = y_train.iloc[:,1]
        y3 = y_train.iloc[:,2]
        y4 = y_train.iloc[:,3]
        y5 = y_train.iloc[:,4]

        y1t = y_test.iloc[:, 0]
        y2t = y_test.iloc[:, 1]
        y3t = y_test.iloc[:, 2]
        y4t = y_test.iloc[:, 3]
        y5t = y_test.iloc[:, 4]

        predictions = []
        for e, (t,tx) in enumerate(zip([y1,y2,y3,y4,y5],[y1t,y2t,y3t,y4t,y5t])):
            model = xgb.XGBRegressor()
            model.fit(x_train, t, eval_set=[(x_test, tx)], early_stopping_rounds=50, verbose=False)
            # select features using threshold
            selection = SelectFromModel(model, threshold=0.005, prefit=True,)
            select_x_train = selection.transform(x_train)
            select_x_test = selection.transform(x_test)
            model = xgb.XGBRegressor()
            model.fit(select_x_train, t.values, eval_set=[(select_x_test, tx.values)], early_stopping_rounds=50, verbose=False)


            # plot_importance(model,max_num_features=50)
            # pyplot.show()
            select_x_public = selection.transform(x_public)
            pred = model.predict(select_x_public)
            if e==0:
                np.save('./select/train/{}.npy'.format(num),arr = select_x_train)
                np.save('./select/test/{}.npy'.format(num), arr=select_x_test)
                np.save('./select/public/{}.npy'.format(num), arr=select_x_public)
                num += 1
            predictions.append(pred[0])
        sample_submission.loc[:, code] = predictions * 2
    # sample_submission.isna().sum().sum()

    columns = list(sample_submission.columns[1:])

    columns = ['Day'] + [str(x).zfill(6) for x in columns]

    sample_submission.columns = columns
    # print(test_submission.iloc[:, 1:].values)
    # print(sample_submission.iloc[:, 1:].values)
    # print(sample_submission.isna().sum().sum())
    print(mean_absolute_error(test_submission.iloc[:, 1:].values, sample_submission.iloc[:, 1:].values))
    print(nmae(test_submission.iloc[:, 1:].values, sample_submission.iloc[:, 1:].values))

# 0.02741967519599727
# 4631.059395323057