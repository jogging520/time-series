from pandas import read_csv
from pandas import datetime
import numpy as np
from math import sqrt
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def parser(x):
	return datetime.strptime(x, '%m/%d/%Y')
# data = pd.read_csv('sp500_data.csv',header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# print(data.head())
# # data.index = data['time'].values
# plt.figure(figsize=(15,5))
# data.plot()
# plt.show()

series = read_csv('sp500_data.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,date_parser=parser)
X = series.values
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]
print(train)
print(test)
input('enter')
history = [x for x in train]
print(test)
input('enter....')
predictions = list()
for t in range(len(test)):
   model = ARIMA(history, order=(1,1,0))
   model_fit = model.fit(disp=0)
   output = model_fit.forecast()
   yhat = output[0]
   predictions.append(yhat)
   obs = test[t]
   history.append(obs)
   # print('predicted=%f, expected=%f ,time=%d' % (yhat, obs, t))
print(predictions)
print(test)
input('enter....')

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# calculate RMSE
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# calulate MAE

mae = mean_absolute_error(test, predictions)
print('Test MAE: %.3f' % mae)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()