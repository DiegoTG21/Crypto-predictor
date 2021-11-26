# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 13:23:35 2021

@author: yeyit
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas import read_csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import LSTM
#from keras.layers import Dense,
from numpy import array

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import os
#hide tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# fix random seed for reproducibility
np.random.seed(36)
from_symbol = 'DOGE'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'hour'
import requests
from datetime import datetime
def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
    return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)
def download_data(from_symbol, to_symbol, exchange, datetime_interval):
    supported_intervals = {'minute', 'hour', 'day'}
    assert datetime_interval in supported_intervals,\
        'datetime_interval should be one of %s' % supported_intervals
    print('Downloading %s trading data for %s %s from %s' %
          (datetime_interval, from_symbol, to_symbol, exchange))
    base_url = 'https://min-api.cryptocompare.com/data/histo'
    url = '%s%s' % (base_url, datetime_interval)
    params = {'fsym': from_symbol, 'tsym': to_symbol,
              'limit': 2000, 'aggregate': 1}
    print(params)
    request = requests.get(url, params=params)
    data = request.json()
    return data
def convert_to_dataframe(data):
    df = pd.json_normalize(data, ['Data'])
    df['datetime'] = pd.to_datetime(df.time, unit='s')
    df = df[['datetime', 'low', 'high', 'open',
             'close', 'volumefrom', 'volumeto']]
    return df
def filter_empty_datapoints(df):
    indices = df[df.sum(axis=1) == 0].index
    print('Filtering %d empty datapoints' % indices.shape[0])
    df = df.drop(indices)
    return df


# convert an array of values into a dataset matrix
def create_dataset(dataset,length=24):
  dataX, dataY = [], []
  for i in range(len(dataset)-length):
    dataX.append(dataset[i])
    dataY.append(dataset[i + 1])
  return np.asarray(dataX), np.asarray(dataY)



# create and fit the LSTM network
def createLSTM(trainX, trainY,n_steps=1, n_features=1,epochNum=20):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(n_steps, n_features),return_sequences=True))
    model.add(LSTM(64,activation='relu',return_sequences=True,dropout=0.2))
    model.add(LSTM(64,activation='relu', input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history=model.fit(trainX, trainY, epochs=epochNum, batch_size=4, verbose=2)
    #save model for later use
    #model.save('./LMST')
    return model,history


def plotPrediction():
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict):len(dataset)-1, :] = testPredict
    #matplotlib.axes.Axes.autoscale(enable=True, axis='both', tight=None)
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.show()
    


def plotDiagnosis(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
# def plotPrediction():
#     # shift train predictions for plotting
#     trainPredictPlot = np.empty_like(dataset)
#     trainPredictPlot[:, :] = np.nan
#     trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict
#     # shift test predictions for plotting
#     testPredictPlot = np.empty_like(dataset)
#     testPredictPlot[:, :] = np.nan
#     testPredictPlot[len(trainPredict):len(dataset)-24, :] = testPredict
#     # plot baseline and predictions
#     plt.plot(testPredict)
#     plt.plot(testPredictPlot)
#     plt.title("2000 hours")
#     plt.show()
   
    
def pltTwo(testPredict,testY):
    plt.figure(figsize=(12,8))
    plt.plot(testY, color='blue', label='Real')
    plt.plot(testPredict, color='red', label='Prediction')
    plt.title('BTC Price Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
# def plotNextHour(hours_for_graph=24,train_size=1600):
#  #get last 24 hours
#     dataPredict = model.predict(dataset)

#     day_dataset=dataset[train_size-hours_for_graph:]
#     day_trainPredict=trainPredict[hours_for_graph]
    
    
#     day_trainPredictPlot=np.empty_like(day_dataset)
#     day_trainPredictPlot[:, :] = np.nan
#     day_trainPredictPlot[1:len(day_trainPredict)+1, :] = day_trainPredict
#     # shift test predictions for plotting
#     day_testPredictPlot = np.empty_like(day_dataset)
#     day_testPredictPlot[:, :] = np.nan
#     day_testPredictPlot[len(day_trainPredict):len(day_dataset)-24, :] = testPredict
#     # plot baseline and predictions
#     plt.plot(scaler.inverse_transform(day_dataset))
#     plt.plot(dataPredict[(len(dataPredict)-24):])
#     plt.title("24 hours and prediction")
#     plt.show()
        
    
    


def graphsFor400and24hours():
    X,y=create_dataset(dataset)
    #Take 80% of data as the training sample and 20% as testing sample
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.20, shuffle=False)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    
    
    model,history=createLSTM(trainX, trainY)
    
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    
    lastDay=dataset[len(dataset)-24:]
    # reshape input to be [samples, time steps, features]
    lastDay_reshaped = np.reshape(lastDay, (lastDay.shape[0], 1, lastDay.shape[1]))
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    lastDayPredict=model.predict(lastDay_reshaped,verbose=0)
    # invert predictions
    trainPredict_rescaled  = scaler.inverse_transform(trainPredict)
    trainY_rescaled = scaler.inverse_transform(trainY)
    testPredict_rescaled  = scaler.inverse_transform(testPredict)
    testY_rescaled  = scaler.inverse_transform(testY)
    
    
    testY = scaler.inverse_transform(testY)
    lastDayPredict_rescaled=scaler.inverse_transform(lastDayPredict)
    lastDay_rescaled=scaler.inverse_transform(lastDay)
    pltTwo(lastDayPredict_rescaled,lastDay_rescaled)
    pltTwo(testPredict_rescaled,testY)
    return model


data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
df = convert_to_dataframe(data)
df = filter_empty_datapoints(df)
df_og=df
#df = read_csv('.BTC_USD_Bitstamp_day_2021-10-14.csv')
df = df_og.iloc[::-1]
df = df.drop(['datetime', 'low', 'high', 'open',
              'volumefrom', 'volumeto'], axis=1)


dataset = df.values

#dataset = dataset.astype('float32')
dataset = dataset.data[::-1] 

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

graphsFor400and24hours()


# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

lists=df.values.tolist()
lists.reverse()
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
lists = scaler.fit_transform(lists)
new_list=[]
testing_list=[]
counter=0
for c in lists:
    if counter<1980:
        new_list.append(c[0])
    else:
        testing_list.append(c[0])
        new_list.append(c[0])

    counter=counter+1

# define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# choose a number of time steps
n_steps = 1
# split into samples
X, y = split_sequence(new_list, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model


model,history=createLSTM(X, y,n_steps, n_features,epochNum=125)
n=int(1974/3)
x_input = array(testing_list[len(testing_list)-n_steps:]).reshape((1, n_steps, n_features))

# fit model
# demonstrate prediction
yhat = model.predict(x_input, verbose=0)
yhat  = scaler.inverse_transform(yhat)

print(yhat)