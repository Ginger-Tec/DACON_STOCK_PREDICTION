from gluonts.dataset.common import ListDataset

from pts.model.deepar import DeepAREstimator
from pts import Trainer
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
# end_data_list = ['20211112']
end_data_list = ['20211001', '20211008', '20211015', '20211022', '20211029', '20211105', '20211112']
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

from gluonts.mx.distribution.neg_binomial import NegativeBinomialOutput,DistributionOutput,Distribution
import gluonts
from pts.modules.distribution_output import PiecewiseLinearOutput,NormalOutput,IndependentNormalOutput,BetaOutput,\
    StudentTMixtureOutput,PiecewiseLinearOutput,PoissonOutput,NegativeBinomialOutput


# gluonts.distribution.distribution_output.DistributionOutput = gluonts.distribution.student_t.StudentTOutput()

for end_date in end_data_list:
    start_weekday = pd.to_datetime(start_date).weekday()
    max_weeknum = pd.to_datetime(end_date).strftime('%V')
    Business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])
    # print('businees', Business_days)
    sample_submission = pd.read_csv(os.path.join(path, sample_name))
    test_submission = copy.deepcopy(sample_submission)
    model = LinearRegression()
    mean_loss =0.
    median_loss =0.
    check1_ = []
    check2_ = []
    for e,code in enumerate(tqdm(stock_list['종목코드'].values)):
        if e==257:
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
            x = data[['Date', 'Close']]
            test_submission.loc[:, code] = pd.concat([x[-5:], data.iloc[-5:]], ignore_index=True)
            x_train = x[:-5]
            # x_train = x[:-35]
            # print(x_train)
            # start = pd.Timestamp("01-04-2016", freq='1D')
            training_data = ListDataset(
                [{"start": x_train.index[0], "target": x_train.Close[:]}],
                freq="1D"
            )


            test_data = ListDataset(
                [{"start": x_train.index[-14], "target": x_train.Close[:]}],
                freq="1D"
            )
            # test_data = ListDataset(
            #     [{"start": x.index[-35-14:-35], "target": x.Close[:]}],
            #     freq="1D"
            # )


            # training_data = ListDataset([{'target': x, 'start': start} for x in x_train],freq='1D')

            estimator = DeepAREstimator(freq="1D", input_size=14, prediction_length=5,trainer=Trainer(epochs=200, device=device, num_batches_per_epoch=100,
                                                                                                      batch_size=2, learning_rate=8e-4,maximum_learning_rate=8e-3),
                                        dropout_rate=0.,
                                        num_parallel_samples=1000,

                                        # distr_output=ImplicitQuantileOutput(output_domain="Positive"),
                                        # distr_output=StudentTMixtureOutput(),
                                        # distr_output= PoissonOutput(),
                                        # distr_output = NegativeBinomialOutput(),
                                        # distr_output=NormalOutput(),
                                        # distr_output = PiecewiseLinearOutput(2)
                                        )
            predictor = estimator.train(training_data=training_data)
            forecast = predictor.predict(test_data)
            mean = 0.
            median = 0.
            for f in forecast:
                mean = f.mean
                median = f.median
                # print(f.mean)
                # print(f.median)
                # print(type(f.mean))
            # print(type(x.iloc[-5:, 1].values))

            tmp1 = nmae(x.iloc[-5:, 1].values, mean)
            tmp2 = nmae(x.iloc[-5:,1].values, median)
            check1_.append(tmp1)
            check2_.append(tmp2)
            # print(tmp1)
            # print(tmp2)
            mean_loss += tmp1
            median_loss += tmp2
    print(check1_)
    print(check2_)
    # print("#####################################")
    # print("#####################################")
    # print("#####################################")
    # print("#####################################")
    # print("#####################################")
    # print(mean_loss / 370)
    # print(median_loss / 370)
    # print("#####################################")
    # print("#####################################")
    # print("#####################################")
    # print("#####################################")
    # print("#####################################")
#3번째 3.29,2.79,2.15,4.6,1.17,0.71,0.91
#3번째 3.75,2.05,1.83,3.0,1.1,0.75,0.90

#1번째 [3.0365027818334602][4.088653716835974][2.571466095590032][1.0091737630972708][0.8738958485907324][0.6342406704528667][1.2403661178659333]
'''
  0%|          | 0/370 [00:00<?, ?it/s]3.2118738576641728
1.723984296550695
100%|██████████| 370/370 [00:00<00:00, 986.55it/s]
100%|██████████| 370/370 [00:00<00:00, 1029.76it/s]
2.5131361729333483
100%|██████████| 370/370 [00:00<00:00, 1392.99it/s]
0.32895613747692165
100%|██████████| 370/370 [00:00<00:00, 1480.16it/s]
0.5969477313474214
100%|██████████| 370/370 [00:00<00:00, 1246.15it/s]
1.0286131936650382
100%|██████████| 370/370 [00:00<00:00, 1246.38it/s]
0.4007996903078169
'''
'''
  0%|          | 1/370 [00:00<01:49,  3.37it/s]3.2118738576641728
2.03120763043415
  1%|          | 3/370 [00:00<01:35,  3.84it/s]3.751234603191607
  1%|          | 4/370 [00:01<01:29,  4.10it/s]1.7491158910375888
4.490517628611287
  2%|▏         | 6/370 [00:01<01:52,  3.24it/s]2.1417730197530784
1.4024399658623965
  2%|▏         | 8/370 [00:02<01:36,  3.75it/s]1.797949632468964
  2%|▏         | 9/370 [00:02<01:39,  3.62it/s]3.4249481042411594
5.849721985114166
  3%|▎         | 11/370 [00:03<01:38,  3.65it/s]2.549622325257789
  3%|▎         | 12/370 [00:03<01:55,  3.09it/s]3.9873829403874685
  4%|▎         | 13/370 [00:03<02:07,  2.79it/s]6.896118630056838
3.375664207368958
  4%|▍         | 15/370 [00:04<01:43,  3.41it/s]2.5434993351581374
4.660909359203667
  5%|▍         | 17/370 [00:04<01:36,  3.67it/s]0.41618167180068577
  5%|▍         | 18/370 [00:05<01:51,  3.15it/s]3.8398390542493694
3.09620428011351
'''
'''

2.5450474004505304
2.5449063775579104

3.74080700674894
3.8115761382252047

5.016742679521302
5.152544196401297



'''