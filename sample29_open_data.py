from urllib.request import  urlopen
from urllib.parse import urlencode, unquote, quote_plus
import urllib
import requests
import pandas as pd
# import xmltodict
import json
# from pandas_datareader import data
from datetime import datetime

key = ''
url = f'http://api.seibro.or.kr/openapi/service/StockSvc/getDividendRankN1?serviceKey={key}&'
queryParams = urlencode({quote_plus('stkTpcd'):'1',quote_plus('listTpcd'):'11',
                         quote_plus('rankTpcd'):'1',quote_plus('year'):'2019',quote_plus('numOfRows'):'1000',
                         quote_plus('pageNo') : '1'})

url2 = url + queryParams
print(url2)
response = urlopen(url2)
results = response.read().decode('utf-8')
print(results)
# results_to_json = xmltodict.parse(results)
# data = json.loads(json.dumps(results_to_json))
# print(data)
# data2 = data['response']['body']['items']['item']
#
# dividends = []
# corp = []
# ticker = []
# for i in data2:
#     dividends.append(i['divAmtPerStk'])
#     corp.append(i['korSecnNm'])
#     ticker.append(i['shotnlsin'])
#
# df = pd.DataFrame([corp, ticker, dividends]).T
# df.columns = ['기업명', '기업코드',' 주당 배당금']
# df['주당 배당금'] = df['주당 배당금'].astype(float).astype(int)
# df2 = df[df['주당 배당금']!=0]
# print(df)
# print(df2)
