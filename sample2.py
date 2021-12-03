import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr

from sklearn.linear_model import LinearRegression
from tqdm import tqdm

path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path,list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))

start_date = '20210104'
end_date = '20211105'

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime('%V')
Business_days = pd.DataFrame(pd.date_range(start_date,end_date,freq='B'), columns = ['Date'])


sample_submission = pd.read_csv(os.path.join(path,sample_name))
print(sample_submission)
model = LinearRegression()
for code in tqdm(stock_list['종목코드'].values):
    data = fdr.DataReader(code, start=start_date, end=end_date)[['Close']].reset_index()
    data = pd.merge(Business_days, data, how='outer')
    data['weekday'] = data.Date.apply(lambda x: x.weekday())
    data['weeknum'] = data.Date.apply(lambda x: x.strftime('%V'))
    data.Close = data.Close.ffill()
    data = pd.pivot_table(data=data, values='Close', columns='weekday', index='weeknum')
    # print(code)
    # print(data.iloc[-1])
    # print(sample_submission.loc[:, code])
    sample_submission.loc[:, code] = pd.concat([data.iloc[-1],data.iloc[-1]],ignore_index=True)
    # print(sample_submission.loc[:, code])
    #44번이 11월 첫째주 데이터

print(sample_submission)
sample_submission.iloc[:,1:] = sample_submission.iloc[:,1:]
#1.028 => 2.8
print(sample_submission)
#     x = data.iloc[0:-2].to_numpy()  # 2021년 1월 04일 ~ 2021년 10월 22일까지의 데이터로
#     y = data.iloc[1:-1].to_numpy()  # 2021년 1월 11일 ~ 2021년 10월 29일까지의 데이터를 학습한다.
#     y_0 = y[:, 0]
#     y_1 = y[:, 1]
#     y_2 = y[:, 2]
#     y_3 = y[:, 3]
#     y_4 = y[:, 4]
#
#     y_values = [y_0, y_1, y_2, y_3, y_4]
#     x_public = data.iloc[-2].to_numpy()  # 2021년 11월 1일부터 11월 5일까지의 데이터를 예측할 것이다.
#
#     predictions = []
#     for y_value in y_values:
#         model.fit(x, y_value)
#         prediction = model.predict(np.expand_dims(x_public, 0))
#         predictions.append(prediction[0])
#     sample_submission.loc[:, code] = predictions * 2
# sample_submission.isna().sum().sum()

columns = list(sample_submission.columns[1:])

columns = ['Day'] + [str(x).zfill(6) for x in columns]

sample_submission.columns = columns

sample_submission.to_csv('./real_sample.csv',index=False)