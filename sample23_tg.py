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
full_data = full_data.fillna(method='bfill')

import torch

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

from pts.model.tempflow import TempFlowEstimator
from pts.model.time_grad import TimeGradEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer

train_grouper = MultivariateGrouper(max_target_dim=full_data.shape[-1])

dataset_train = train_grouper(dataset.train)
dataset_test = test_grouper(dataset.test)
temp = next(iter(dataset_train))
print(temp['target'].shape)