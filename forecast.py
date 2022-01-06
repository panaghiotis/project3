import argparse
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# 1 time series default
num = 1

# ----------------------INPUT PARSER for python --------------------------------------------------
# by https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
# Required dataset argument
parser.add_argument('-d', help='A required string positional argument')
# Optional argument
parser.add_argument('-n', type=int, help='An optional integer argument')
args = parser.parse_args()
if args.n != None:
    num = args.n
dataset = args.d
if args.d == None:
    dataset = input("Enter a dataset filepath: ")
# ------------------------------------------------------------------------------------------------

df=pd.read_csv(dataset, sep = '\t')

# get number of columns and add column keys
col_names = []
col_size = len(df.columns) - 1
for n in range(col_size):
    col_names.append(str(n))

# read csv file again and transpose it
df=pd.read_csv(dataset, sep = '\t', names = col_names)
df = pd.DataFrame.transpose(df)
# print("Number of rows and columns:", df.shape)
# print(df.head(2))

# keep time series names for plots later
list_names = list(df.columns)

# train almost 80% of data
data_to_be_trained = (80 * len(df.index)) // 100

# TODO: random time series. Make a list of random numbers and traverse the list with for i puting i as time series:time series +1
print("Computing different models for each time series...")
# for each time series
for time_series in range(num):

    training_set = df.iloc[:data_to_be_trained, time_series:(time_series+1)].values
    # print(training_set)

    # predict almost 20% of data
    test_set = df.iloc[data_to_be_trained:, time_series:(time_series+1)].values
    # print(test_set)

    # Feature Scaling
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 time-steps and 1 output
    X_train = []
    y_train = []
    for i in range(60, data_to_be_trained):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    # (740, 60, 1)

    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 5, batch_size = 32)

    # Prepare the test data
    # Getting the predicted value
    dataset_train = df.iloc[:data_to_be_trained, time_series:(time_series+1)]
    dataset_test = df.iloc[data_to_be_trained:, time_series:(time_series+1)]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_test.shape)
    # (459, 60, 1)

    # Make predictions
    predicted_value = model.predict(X_test)
    print("------------------------------------------------------------------")
    print(predicted_value.shape)
    predicted_value = sc.inverse_transform(predicted_value)

    # Visualising the results
    plt.plot(col_names[data_to_be_trained:], dataset_test.values, color = "red", label = "Real Value")
    plt.plot(col_names[data_to_be_trained:], predicted_value, color = "blue", label = "Predicted Value")
    plt.xticks(np.arange(0, 146, 60))
    title = 'Time Series '
    for ts_name in list_names[time_series:(time_series+1)]:
        title = title + ts_name + ' Value Prediction'
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
print("Done!")

print("Computing one model for all time series...")
# for n time series
sc_list = []
for i in range(len(df.columns)):
    sc_list.append(MinMaxScaler(feature_range = (0, 1)))
sc = np.array(sc_list)

X_train = []
y_train = []
for time_series in range(len(df.columns)):
    training_set = df.iloc[:data_to_be_trained, time_series:(time_series+1)].values
    test_set = df.iloc[data_to_be_trained:, time_series:(time_series+1)].values

    # Feature Scaling
    # sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc[time_series].fit_transform(training_set)

    # Creating a data structure with 60 time-steps and 1 output
    for i in range(60, data_to_be_trained):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# (740, 60, 1)

model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units=50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units=1))

# Compiling the RNN
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs=1, batch_size=32)

# predict and visualise for each time series
for time_series in range(num):
    # Prepare the test data
    # Getting the predicted value
    dataset_train = df.iloc[:data_to_be_trained, time_series:(time_series+1)]
    dataset_test = df.iloc[data_to_be_trained:, time_series:(time_series+1)]
    dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc[time_series].transform(inputs)
    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(X_test.shape)
    # (459, 60, 1)

    # Make predictions
    predicted_value = model.predict(X_test)
    predicted_value = sc[time_series].inverse_transform(predicted_value)

    # Visualising the results
    plt.plot(col_names[data_to_be_trained:],dataset_test.values, color = "red", label = "Real Value")
    plt.plot(col_names[data_to_be_trained:],predicted_value, color = "blue", label = "Predicted Value")
    plt.xticks(np.arange(0,146,60))
    title = 'Time Series '
    for ts_name in list_names[time_series:(time_series+1)]:
        title = title + ts_name + ' Value Prediction'
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
print("Done!")
