import pandas as pd
import numpy as np
import os
import FinanceDataReader as fdr
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from tqdm import tqdm
from sklearn.model_selection import KFold


path = './'
list_name = 'stock_list.csv'
stock_list = pd.read_csv(os.path.join(path,list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x : str(x).zfill(6))

start_date = '20210104'
end_date = '20211105'

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime('%V')
Business_days = pd.DataFrame(pd.date_range(start_date,end_date,freq='B'), columns = ['Date'])

sample_name = 'sample_submission.csv'
sample_submission = pd.read_csv(os.path.join(path,sample_name))


from typing import Tuple

def NMAE(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    y = dtrain.get_label()
    return 100*np.absolute(np.subtract(predt,y)).mean()

kf = KFold(n_splits=6)


def Model(estimator, model_type):
    for code in tqdm(stock_list['종목코드'].values):
        data = fdr.DataReader(code, start=start_date, end=end_date)[['Close']].reset_index()
        data = pd.merge(Business_days, data, how='outer')
        data['weekday'] = data.Date.apply(lambda x: x.weekday())
        data['weeknum'] = data.Date.apply(lambda x: x.strftime('%V'))
        data.Close = data.Close.ffill()
        data = pd.pivot_table(data=data, values='Close', columns='weekday', index='weeknum')

        #
        x = data.iloc[0:-2].to_numpy()
        y = data.iloc[1:-1].to_numpy()

        # pycaret data

        y_0 = y[:, 0]
        y_1 = y[:, 1]
        y_2 = y[:, 2]
        y_3 = y[:, 3]
        y_4 = y[:, 4]

        y_values = [y_0, y_1, y_2, y_3, y_4]
        x_public = data.iloc[-2].to_numpy()
        x_private = data.iloc[-1].to_numpy()

        public_predictions = []
        private_predictions = []
        for y_value in y_values:
            public_folds = []
            private_folds = []
            for id, (train_id, valid_id) in enumerate(kf.split(x)):
                x_train, y_train = x[train_id], y_value[train_id]
                x_valid, y_valid = x[valid_id], y_value[valid_id]
                model = estimator
                if model_type == 'linear':
                    model.fit(x_train, y_train)
                elif model_type == 'boost':
                    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=50, verbose=False)
                public_prediction = model.predict(np.expand_dims(x_public, 0))
                private_prediction = model.predict(np.expand_dims(x_private, 0))
                public_folds.append(public_prediction[0])
                private_folds.append(private_prediction[0])
            public_prediction_folds = np.mean(np.array(public_folds), axis=0)
            private_prediction_folds = np.mean(np.array(private_folds), axis=0)
            public_predictions.append(public_prediction_folds)
            private_predictions.append(private_prediction_folds)
        sample_submission.loc[:, code] = public_predictions + private_predictions

    columns = list(sample_submission.columns[1:])

    columns = ['Day'] + [str(x).zfill(6) for x in columns]
    sample_submission.columns = columns
    return sample_submission

linear = LinearRegression()
linear_col = Model(linear,'linear')
# xgb = XGBRegressor(n_estimators=1000, max_depth=3, eta=0.1, subsample=0.5, colsample_bytree=0.6, obj=NMAE,
#                                  feval=NMAE, verbosity=0, silent=0,verbose_eval=None)
# xgb_col = Model(xgb,'boost')
ela = ElasticNet(alpha=0.02, max_iter=1000000, normalize=True, l1_ratio = 0.8)
ela_col = Model(ela,'linear')

import copy

submission = copy.deepcopy(linear_col)
submission.iloc[:,1:] = (linear_col.iloc[:,1:]+ela_col.iloc[:,1:])/2
submission.to_csv('./my_baseline2.csv',index=False)
