import json
import numpy as np
import io
with open(r'C:\Users\Administrator\.mxnet\gluon-ts\datasets\electricity_nips\train\data.json', 'rb') as f:
    fbuf = io.BufferedReader(f)
    for i in range(10):
        tmp = fbuf.readline()
        print(type(tmp))
        print(tmp)

with open('./sample.txt', 'wb') as file:
    # data = {"k": 2312, "a": 2342}
    data = b'{"k": 2312, "a": 2342}'
    file.write(data)

with open('./sample.txt', 'rb') as f:
    fbuf = io.BufferedReader(f)
    tmp = fbuf.readline()
    print(tmp)
# print(json.dumps(json_data))