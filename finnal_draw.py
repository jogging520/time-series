import numpy as np
from math import sqrt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#
def parser(x):
	return datetime.strptime(x, '%m/%d/%Y')
#
# series = pd.read_csv('sp500_data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)
#
# X = series.values
# size = int(len(X) * 0.7)
# train, test = X[0:size], X[size:len(X)]
# print(len(test))
# savetest= pd.DataFrame(test)
# savetest.to_csv('test.csv')
#data = pd.read_csv('final.csv',usecols=[0,4,8])
#data = pd.read_csv('final.csv',usecols=[1,5,9])
#data = pd.read_csv('final.csv',usecols=[2,6,10])
#data = pd.read_csv('final.csv',usecols=[3,7,11])
TEST =  pd.read_csv('final.csv',usecols=[0,1],header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)

data1 = pd.read_csv('final.csv',usecols=[0,5],header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)
data2 = pd.read_csv('final.csv',usecols=[0,9],header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)
data3 = pd.read_csv('final.csv',usecols=[0,13],header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)
print(data1.head())

plt.plot(TEST,  label="Test data")
plt.plot(data1, label="ARIMA_14_days" )
plt.plot(data2, label="BP_14_days" )
plt.plot(data3, label="LSTM_14_days" )
plt.legend(loc='upper left')
plt.ylabel('S&P500_index')
plt.xlabel('Trade_date')
plt.show()
