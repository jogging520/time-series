import numpy as np
from math import sqrt
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def parser(x):
	return datetime.strptime(x, '%m/%d/%Y')

series = pd.read_csv('sp500_data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)
X = series.values
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]

#data = pd.read_csv('final.csv',usecols=[0,4,8])
#data = pd.read_csv('final.csv',usecols=[1,5,9])
#data = pd.read_csv('final.csv',usecols=[2,6,10])
data = pd.read_csv('final.csv',usecols=[3,7,11])

plt.plot(test)
data.plot()
plt.show()

print(data)