# print("---결과---")
# print("")
# print("test-sample1: 5.587857620228853")
# print("")
# print("test-sample2: 6.707557631014269")
# print("")
# print("test-sample3: 5.22106559142984")
# print("")
# print("test-sample4: 5.028371839875273")
# print("")
# print("test-sample5: 4.356331186744514")
# print("")
# print("test-sample6: 4.445121258178799(public_score)")
# print("")
# print("test-sample7: 5.361476388836757")
# print(1251.1432037456173/370)
# print(1251.3132674667786/370)


import pandas as pd
import numpy as np
res = pd.read_csv('./result.csv')
pre = pd.read_csv('./fma_presub.csv')
test = pd.read_csv('./fma_submission.csv')
test2 = pd.read_csv('./fma_submission2.csv')
real = pd.read_csv('./real_sample.csv')
arima1 = pd.read_csv('./just_Arima.csv')
arima2 = pd.read_csv('./public_private_Arima.csv')
arima3 = pd.read_csv('./nyear_Arima.csv')
arima4 = pd.read_csv('./col_change_Arima.csv')
arima5 = pd.read_csv('./store1_Arima.csv')
arima6 = pd.read_csv('./storeall_Arima.csv')
arima7 = pd.read_csv('./store_return_Arima.csv')
arima8 = pd.read_csv('./not_year_Arima.csv')
arima9 = pd.read_csv('./2_Arima.csv')
arima10 = pd.read_csv('./3_Arima.csv')
arima11 = pd.read_csv('./re_Arima.csv')
arima12 = pd.read_csv('./all_Arima.csv')
print(test.all()==test2.all())
def nmae(true, pred):
    score = np.mean(np.abs(true - pred) / true) * 100
    return score

print(real)

print(nmae(real.iloc[:5,1:].values.reshape(-1),arima12.iloc[:5,1:].values.reshape(-1)))
# print(pre.iloc[:5,2].values.reshape(-1),test.iloc[:5,2].values.reshape(-1))