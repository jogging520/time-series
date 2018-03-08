from random import randint
from math import sqrt
from numpy import array
import numpy as np
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as  pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from numpy import concatenate
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


# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
	X1, X2, y = list(), list(), list()
	for _ in range(n_samples):
		# generate source sequence
		source = generate_sequence(n_in, cardinality)
		# define padded target sequence
		target = source[:n_out]
		target.reverse()
		# create padded input target sequence
		target_in = [0] + target[:-1]
		# encode
		src_encoded = to_categorical([source], num_classes=cardinality)
		tar_encoded = to_categorical([target], num_classes=cardinality)
		tar2_encoded = to_categorical([target_in], num_classes=cardinality)
		# store
		X1.append(src_encoded)
		X2.append(tar2_encoded)
		y.append(tar_encoded)
	return array(X1), array(X2), array(y)

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output)
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input the size of test
	target_seq = array([0.0 for _ in range(n_steps)]).reshape(1, 1, n_steps)
	# collect predictions
	output = list()
	for t in range(1):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)


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

# split into train and test sets
values = reframed.values

n_train_days =2983 # train is time_step0%
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-time_step], train[:,-time_step:]
test_X, test_y = test[:, :-time_step], test[:, -time_step:]
X2 = train_y[:,0:time_step-1]
b= np.ones(len(train_y))
X2= np.insert(X2, 0, values=b, axis=1)

#X2 = X2[np.newaxis,:]
# print(X2.shape)
# print(X2)
# print('y:-1',train_y[:,0:time_step-1])
# input('enterx2..')
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))

X2 =X2.reshape((X2.shape[0], 1, X2.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
input('enter...')
# configure problem
n_steps_in = time_step
n_steps_out = time_step
# define model

train, infenc, infdec = define_models(time_step, time_step, 128)
train.compile(optimizer='adam', loss='mae')
X1, y= train_X, train_y

# train model
train.fit([X1, X2], y, epochs=2)

# make a prediction
# yhat = predict_sequence(infenc, infdec, test_X, n_steps_out, time_step)
# print('yhat:',yhat.shape)

predictions = list()
for t in range(test_X.shape[0]):
	# temp = test_X[t]
	# temp = temp[np.newaxis, :, :]
    yhat = predict_sequence(infenc, infdec, test_X[t][np.newaxis, :, :], n_steps_out, time_step)
    predictions.append(yhat)
print(array(predictions).shape)
predictions = array(predictions)
predictions=predictions.reshape(predictions.shape[0],predictions.shape[2])
# invert scaling for forecast
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_yhat = concatenate((predictions, test_X[:, time_step:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0:time_step]


save = pd.DataFrame(inv_yhat)
save.to_csv('seq2seq_step3.csv')
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