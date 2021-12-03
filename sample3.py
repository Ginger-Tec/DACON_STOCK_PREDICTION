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
def nmae(true,pred):
    score = np.mean(np.abs(true-pred) / true) * 100
    return score

path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path,list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))

# start_date = '20210104'
start_date = '20200601'
# end_data_list = ['20211001','20211008','20211015','20211022','20211029','20211105','20211112']
end_data_list = ['20211105']
temp = []
sample_submission = pd.read_csv(os.path.join(path, sample_name))
test_submission = copy.deepcopy(sample_submission)
for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])



    for code in tqdm(stock_list['종목코드'].values):
        data = fdr.DataReader(code, start=start_date, end=end_date)[['Close']].reset_index()
        data = pd.merge(Business_days, data, how='outer')
        data['weekday'] = data.Date.apply(lambda x: x.weekday())
        data['weeknum'] = data.Date.apply(lambda x: x.strftime('%V'))
        data.Close = data.Close.ffill()
        data = pd.pivot_table(data=data, values='Close', columns='weekday', index='weeknum')
        # print(data.isna().sum())
        # if data.any().isna() == True:
        #     print(data)
        #     break
        data = data.dropna(axis=0)

        test_submission.loc[:, code] = pd.concat([data.iloc[-1], data.iloc[-1]], ignore_index=True)
        continue
    # temp.append(test_submission)
    #     x = data.iloc[0:-2].to_numpy()  # 2021년 1월 04일 ~ 2021년 10월 22일까지의 데이터로
    #     y = data.iloc[1:-1].to_numpy()  # 2021년 1월 11일 ~ 2021년 10월 29일까지의 데이터를 학습한다.
    #
    #     y_0 = y[:, 0]
    #     y_1 = y[:, 1]
    #     y_2 = y[:, 2]
    #     y_3 = y[:, 3]
    #     y_4 = y[:, 4]
    #
    #     y_values = [y_0, y_1, y_2, y_3, y_4]
    #     x_public = data.iloc[-2].to_numpy()
    #     model = LinearRegression()
    #     predictions = []
    #     for y_value in y_values:
    #         model.fit(x, y_value)
    #         prediction = model.predict(np.expand_dims(x_public, 0))
    #         predictions.append(prediction[0])
    #     sample_submission.loc[:, code] = predictions * 2
    # sample_submission.isna().sum().sum()
    #
    # columns = list(sample_submission.columns[1:])
    #
    # columns = ['Day'] + [str(x).zfill(6) for x in columns]
    #
    # sample_submission.columns = columns
    # print(nmae(test_submission.iloc[:,1:].values,sample_submission.iloc[:,1:].values))
print(test_submission)
test_submission.to_csv('./result.csv',index=False)
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
