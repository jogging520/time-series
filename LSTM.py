from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as  pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def plot_results(predicted_data, true_data):
    fig = pyplot.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    pyplot.plot(predicted_data, label='Prediction')
    pyplot.legend()
    pyplot.show()


# load dataset
dataset = read_csv('SP500_data.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

time_step = 3
# frame as supervised learning
reframed = series_to_supervised(scaled,time_step,time_step)

print(reframed.head())

input('enter')

# split into train and test sets
values = reframed.values
print(values)
print('len',len(values))
n_train_days =2983 # train is time_step0%
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-time_step], train[:,-time_step:]
test_X, test_y = test[:, :-time_step], test[:, -time_step:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
input('enter')
# design network
model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(time_step))
model.compile(loss='mae', optimizer='adam')
# fit network

history = model.fit(train_X, train_y, epochs=10, batch_size=14, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
#pyplot.show()

# make a prediction
yhat = model.predict(test_X)
print('yhat:',yhat.shape)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, time_step:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0:time_step]

save = pd.DataFrame(inv_yhat)
save.to_csv('clstm_step14.csv')
predictions = list()
# for t in range(test_X.shape[0]):
#     for yhat_item in inv_yhat[t]:
#         predictions.append(yhat_item)
#     print(t)
#     t=t+3
print(len(predictions))
# invert scaling for actual
test_y = test_y.reshape((len(test_y), time_step))
print('inv_yhat is:',inv_yhat)
inv_y = concatenate((test_y, test_X[:, time_step:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0:time_step]
print('inv_y:',inv_y)
print('inv_yhat is:',inv_yhat.shape)
print('inv_y size:',inv_y.shape)

plot_results(inv_yhat,inv_y)
input('enter')
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y,inv_yhat))
print('Test RMSE: %.3f' % rmse)

error = mean_squared_error(inv_y,inv_yhat)
print('Test MSE: %.3f' % error)
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y,inv_yhat))
print('Test RMSE: %.3f' % rmse)
# calulate MAE

mae = mean_absolute_error(inv_y,inv_yhat)
print('Test MAE: %.3f' % mae)
#pyplot.plot(inv_yhat)
#pyplot.show()