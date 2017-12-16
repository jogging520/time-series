import numpy as np
from math import sqrt
import pandas as pd
import pyflux as pf
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#matplotlib inline
#存在日期转换的问题----------------------------------
def parser(x):
	return datetime.strptime(x, '%Y/%m/%d')

# data = pd.read_csv('sp500.csv',header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# print(data.head())
# # data.index = data['time'].values
# plt.figure(figsize=(15,5))
# data.plot()
# plt.show()

series = pd.read_csv('sp500.csv')
# plt.ylabel('price')
# plt.title('time')
# series.plot()
# plt.show()

model = pf.ARIMA(data=series, ar=4, ma=4, target='price', family=pf.Normal())

x = model.fit("MLE")
# x.summary()
# #
# model.plot_fit(figsize=(15,10))
# model.plot_predict_is(h=20, figsize=(15,5))

yhat = model.predict_is(20)
y = series['price'].values[-20:]

print(y)
input('enter')
# calculate RMSE
rmse = sqrt(mean_squared_error(y, yhat))
print('Test RMSE: %.3f' % rmse)