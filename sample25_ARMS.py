import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
import copy
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import xgboost as xgb
import warnings

from statsmodels.tsa.arima.model import ARIMA

import statsmodels.api as sm
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


def nmae(true, pred):
    score = np.mean(np.abs(true - pred) / true) * 100
    return score


from sklearn.metrics import mean_absolute_error
import pywt
path = './'
list_name = 'stock_list.csv'
sample_name = 'sample_submission.csv'

stock_list = pd.read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))

# start_date = '20210104'
# start_date = '20160104'
start_date = '20140104'
end_data_list = ['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']
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
    total_loss =0.
    for e, code in enumerate(tqdm(stock_list['종목코드'].values)):
        if e == 0:
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

            x_public = data.iloc[-5:]
            x_public = x_public['Close']
            data = data.iloc[:-5]
        #     print()
        #     close = data['Close']
        #     high = data['High']
        #     low = data['Low']
        #     open = data['Open']
        #     week = data['week']
        #
        #     # test_submission.loc[:, code] = pd.concat([x[-5:], data.iloc[-5:]], ignore_index=True)
        #     collect = []
        #     label = []
        #     for yi in range(1,101):
        #         x_close = close[:-5*yi]
        #         x_high = high[:-5*yi]
        #         x_low = low[:-5*yi]
        #         x_open = open[:-5*yi]
        #         price = []
        #         for x_train in [x_close, x_high, x_low, x_open]:
        #             model = ARIMA(x_train, order=(1, 1, 0))
        #             model_fit = model.fit()
        #             forecast_close = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
        #             price.append(forecast_close)
        #         collect.append(np.concatenate(price))
        #         if yi == 1:
        #             label.append(close[-5 * yi:])
        #         else:
        #             label.append(close[-5*yi:-5*(yi-1)])
        #     lin_data = np.stack(collect,axis=0)
        #     lin_label =np.stack(label,axis=0)
        #
        #     label1 = lin_label[:,0]
        #     label2 = lin_label[:,1]
        #     label3 = lin_label[:,2]
        #     label4 = lin_label[:,3]
        #     label5 = lin_label[:,4]
        #
        #     lin_labels = [label1, label2, label3, label4, label5]
        #     predictions = []
        #     for labels in lin_labels:
        #         lin_model = LinearRegression()
        #         lin_model.fit(lin_data,labels)
        #
        #         price = []
        #         for x_train in [close, high, low, open]:
        #             model = ARIMA(x_train, order=(1, 1, 0))
        #             model_fit = model.fit()
        #             forecast_close = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
        #             price.append(forecast_close)
        #         x_price = np.concatenate(price).reshape(1, 20)
        #         prediction = lin_model.predict(x_price)
        #         predictions.append(prediction[0])
        #     predictions = np.array(predictions)
        #     print(predictions.shape)
        #     tmp = nmae(x_public, predictions.reshape(-1))
        #     print(tmp)
        # total_loss += tmp
    # print(total_loss / 370)
    #     sample_submission.loc[:, code] = forecast_data.tolist() * 2
    #     print(sample_submission.loc[:, code].values)
    #     print(nmae(x[-5:], np.exp(forecast_data)))
    #     total_loss += nmae(close[-5:], np.exp(forecast_data))
    # print(total_loss/370)
        x = data[:-5]
        test_submission.loc[:, code] = pd.concat([x[-5:], data.iloc[-5:]], ignore_index=True)
        [(cA1_train, cD1_train)] = pywt.swt(np.ravel(x[5:-5]), 'db15', level=1)
        data_pad = np.lib.pad(np.ravel(x[5:-5]), (0, 298), 'constant', constant_values=0)
        [(cA1, cD1)] = pywt.swt(data_pad, 'db15', level=1)
        # cA1_re = pywt.iswt([cA1_train,cD1_train], 'db3')
        model = ARIMA(cA1_train, order=(1, 1, 0))
        model_fit = model.fit()
        forecast_data = model_fit.forecast(steps=5)  # 마지막 5일의 예측 데이터
        # sample_submission.loc[:, code] = forecast_data.tolist() * 2
        cA1_re = pywt.iswt([forecast_data, np.zeros(4)], 'db1')
        print(nmae(x[-5:], cA1_re))
        # total_loss += nmae(x[-5:], forecast_data)
    # print(total_loss/370)

#첫번째
'''
2.782184318989155

1.6266461616301657

2.604485752351592

0.604667555372254

0.7645958601941526

1.2773911562059352

0.436027921902351
'''

#비교대상
'''
  0%|          | 1/370 [00:00<01:49,  3.37it/s]3.2118738576641728
2.03120763043415
  1%|          | 3/370 [00:00<01:33,  3.91it/s]3.751234603191607
  1%|          | 4/370 [00:01<01:30,  4.05it/s]1.7491158910375888
  1%|▏         | 5/370 [00:01<01:38,  3.69it/s]4.490517628611287
2.1417730197530784
  2%|▏         | 7/370 [00:02<01:48,  3.33it/s]1.4024399658623965
  2%|▏         | 8/370 [00:02<01:39,  3.65it/s]1.797949632468964
3.4249481042411594
  3%|▎         | 10/370 [00:02<01:40,  3.60it/s]5.849721985114166
'''
#arms
'''
D:\python_folder\python.exe D:/Administrator/dacon_binance/sample25_ARMS.py
  0%|          | 1/370 [00:21<2:14:36, 21.89s/it]2.8474906391084924
  1%|          | 2/370 [00:42<2:08:30, 20.95s/it]1.8030517110164106
  1%|          | 3/370 [01:09<2:27:01, 24.04s/it]3.1872727842470012
0.9701564506569206
  1%|▏         | 5/370 [01:55<2:23:36, 23.61s/it]5.339902797797167
2.039141904140166
  2%|▏         | 7/370 [02:44<2:29:06, 24.65s/it]1.3945861691929424
  2%|▏         | 8/370 [03:13<2:36:38, 25.96s/it]3.11426314566941
4.885665989998631
  3%|▎         | 10/370 [04:08<2:40:44, 26.79s/it]9.368264077630986
1.5483445141558239
'''

'''ewm
D:\python_folder\python.exe D:/Administrator/dacon_binance/sample25_ARMS.py
100%|██████████| 370/370 [01:35<00:00,  3.88it/s]
3.98862434783635
100%|██████████| 370/370 [01:34<00:00,  3.92it/s]
3.2423274307834093
100%|██████████| 370/370 [01:35<00:00,  3.89it/s]
2.3086535463047646
100%|██████████| 370/370 [01:34<00:00,  3.93it/s]
2.5408512065445543
100%|██████████| 370/370 [01:34<00:00,  3.93it/s]
2.694790465984659
100%|██████████| 370/370 [01:34<00:00,  3.94it/s]
2.8516902970059914
100%|██████████| 370/370 [01:34<00:00,  3.91it/s]
3.4695004278138075
'''
#dewm
'''
100%|██████████| 370/370 [01:33<00:00,  3.96it/s]
3.8097276627278043
100%|██████████| 370/370 [01:32<00:00,  3.98it/s]
3.0969025929246423
100%|██████████| 370/370 [01:33<00:00,  3.95it/s]
2.1959634007387634
100%|██████████| 370/370 [01:33<00:00,  3.97it/s]
2.442473380603756
100%|██████████| 370/370 [01:34<00:00,  3.90it/s]
2.5794294872703523
100%|██████████| 370/370 [01:33<00:00,  3.96it/s]
2.7481499403346685
100%|██████████| 370/370 [01:35<00:00,  3.86it/s]
3.3355056341913483
'''

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