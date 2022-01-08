import argparse
import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
import math

# load model training or run model training
loadModel = True


# ----------------------INPUT PARSER for python --------------------------------------------------
# by https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
# Required dataset argument
parser.add_argument('-d', help='A required string positional argument')
# Required query dataset argument
parser.add_argument('-q', help='A required string positional argument')
# Required output dataset argument
parser.add_argument('-od', help='A required string positional argument')
# Required output dataset argument
parser.add_argument('-oq', help='A required string positional argument')
# Optional argument
parser.add_argument('-load', type=int, help='An optional integer argument')
args = parser.parse_args()

dataset = args.d
if args.d == None:
    dataset = input("Enter a dataset filepath: ")

q_dataset = args.q
if args.q == None:
    q_dataset = input("Enter a query dataset filepath: ")

output_file = args.od
if args.od == None:
    output_file = input("Enter an output file name: ")
if output_file[-4:] != '.csv':
    output_file = output_file + '.csv'

output_q = args.oq
if args.oq == None:
    output_q = input("Enter an output query file name: ")
if output_q[-4:] != '.csv':
    output_q = output_q + '.csv'

if args.load == 0:
    loadModel = False
# ------------------------------------------------------------------------------------------------

# -----------------------------------------------Parameters---------------------------------------
window_length = 10
encoding_dim = 3
epochs = 100
# test_samples = 136
test_samples = 365
# ------------------------------------------------------------------------------------------------

# -----------------------------------------------Utils--------------------------------------------

def plot_examples(stock_input, stock_decoded):
    n = 10
    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(list(np.arange(0, test_samples, 37))):
        # display original
        ax = plt.subplot(2, n, i + 1)
        if i == 0:
            ax.set_ylabel("Input", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_input[idx])
        ax.get_xaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        if i == 0:
            ax.set_ylabel("Output", fontweight=600)
        else:
            ax.get_yaxis().set_visible(False)
        plt.plot(stock_decoded[idx])
        ax.get_xaxis().set_visible(False)
    plt.show()

def plot_history(history):
    plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"])
    plt.title("Train loss")
    ax = plt.subplot(1, 2, 2)
    plt.plot(history.history["val_loss"])
    plt.title("Test loss")
    plt.show()

# ------------------------------------------------------------------------------------------------

# ---------------------------------------Read and transpose csv files-----------------------------
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
# print(len(df.columns))
# print(len(df.index))
# ------------------------------------------------------------------------------------------------

# ---------------------------------------Scaling--------------------------------------------------
# train almost 80% of data
data_to_be_trained = (80 * len(df.columns)) // 100
print(data_to_be_trained)

sc_list = []
for i in range(len(df.columns)):
    sc_list.append(MinMaxScaler(feature_range = (0, 1)))
sc = np.array(sc_list)

# test_samples = (len(df.index) - data_to_be_trained - window_length) * len(df.columns)

X_train, X_test = [], []
#for time_series in range(1):
for time_series in range(data_to_be_trained):
    training_set = df.iloc[:, time_series:(time_series+1)].values
    # print(training_set)

    # Feature Scaling
    # sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc[time_series].fit_transform(training_set)

    # Creating a data structure with window length time-steps and 1 output
    for i in range(math.ceil(len(df.index) / window_length)):
        if i == 0:
            X_train.append(training_set_scaled[0:window_length, 0])
        else:
            if i == math.ceil(len(df.index) / window_length) - 1:
                # X_train.append(training_set_scaled[(i * window_length):, 0])
                lastWindow_len = len(training_set_scaled[(i*window_length):, 0])
                zero_list = []
                for j in range(lastWindow_len, window_length):
                    zero_list.append(0)
                last_window = np.append(training_set_scaled[(i * window_length):, 0], zero_list)
                # print(last_window)
                X_train.append(last_window)
            else:
                X_train.append(training_set_scaled[(i*window_length):((i+1)*window_length), 0])

#print(X_train)
X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

for time_series in range(data_to_be_trained, len(df.columns)):
    # predict almost 20% of data
    test_set = df.iloc[:, time_series:(time_series + 1)].values
    # sc = MinMaxScaler(feature_range=(0, 1))
    test_set_scaled = sc[time_series].fit_transform(test_set)

    final_data = math.ceil(len(test_set_scaled) / window_length)
    for i in range(final_data):
        if i == 0:
            X_test.append(test_set_scaled[0:window_length, 0])
        else:
            if i == final_data - 1:
                # X_test.append(test_set_scaled[(i*window_length):len(test_set_scaled), 0])
                lastWindow_len = len(test_set_scaled[(i * window_length):, 0])
                zero_list = []
                for j in range(lastWindow_len, window_length):
                    zero_list.append(0)
                last_window = np.append(test_set_scaled[(i * window_length):, 0], zero_list)
                # print(last_window)
                X_test.append(last_window)
            else:
                X_test.append(test_set_scaled[(i * window_length):((i + 1) * window_length), 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


print(X_train.shape)
print(X_test.shape)
# ------------------------------------------------------------------------------------------------

# --------------------------------------1D Convolutional autoencoder------------------------------
input_window = Input(shape=(window_length,1))
x = Conv1D(16, 3, activation="relu", padding="same")(input_window) # 10 dims
#x = BatchNormalization()(x)
x = MaxPooling1D(2, padding="same")(x) # 5 dims
x = Conv1D(1, 3, activation="relu", padding="same")(x) # 5 dims
#x = BatchNormalization()(x)
encoded = MaxPooling1D(2, padding="same")(x) # 3 dims

encoder = Model(input_window, encoded)

if loadModel is True:
    autoencoder = tf.keras.models.load_model('saved_models/model_reduce')

    # Check its architecture
    autoencoder.summary()

else:
    # 3 dimensions in the encoded layer
    x = Conv1D(1, 3, activation="relu", padding="same")(encoded) # 3 dims
    #x = BatchNormalization()(x)
    x = UpSampling1D(2)(x) # 6 dims
    x = Conv1D(16, 2, activation='relu')(x) # 5 dims
    #x = BatchNormalization()(x)
    x = UpSampling1D(2)(x) # 10 dims
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x) # 10 dims
    autoencoder = Model(input_window, decoded)
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    history = autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=1024,
                    shuffle=True,
                    validation_data=(X_test, X_test))

    # Save the entire model as a SavedModel.
    autoencoder.save('saved_models/model_reduce')

    plot_history(history)

list_out = []
for time_series in range((len(df.columns) - data_to_be_trained)):

    decoded_stocks = autoencoder.predict(X_test[(time_series*(len(X_test) // (len(df.columns) - data_to_be_trained))):((time_series+1)*(len(X_test) // (len(df.columns) - data_to_be_trained)))])
    # print(decoded_stocks.shape)
    # plot_history(history)
    if time_series < 1:
        plot_examples(X_test[(time_series*(len(X_test) // (len(df.columns) - data_to_be_trained))):((time_series+1)*(len(X_test) // (len(df.columns) - data_to_be_trained)))], decoded_stocks)
    # print(decoded_stocks.shape)
    #nsamples, nx, ny = decoded_stocks.shape
    #d2_decoded_stocks = decoded_stocks.reshape((nsamples, nx * ny))
    # print(d2_decoded_stocks.shape)
    # d2_decoded_stocks = sc.inverse_transform(d2_decoded_stocks)
    # print(d2_decoded_stocks.shape)
    # print(d2_decoded_stocks)

    # reduced_encoded = encoder.predict(X_test[(time_series*(len(X_test) // len(df.columns))):((time_series+1)*(len(X_test) // len(df.columns)))])
    # print(reduced_encoded.shape)
for time_series in range(len(df.columns)):
    if time_series < data_to_be_trained:
        trained = X_train[(time_series*(len(X_train) // data_to_be_trained)):((time_series+1)*(len(X_train) // data_to_be_trained))]
        reduced_encoded = encoder.predict(trained)
        nsamples, nx, ny = reduced_encoded.shape
        d2_reduced_encoded = reduced_encoded.reshape((nsamples, nx * ny))
        d2_reduced_encoded = sc[time_series].inverse_transform(d2_reduced_encoded)
        # print(d2_reduced_encoded.shape)
        # print(d2_reduced_encoded)
        reduced_time_series = []
        for i in range(len(d2_reduced_encoded)):
            for j in range(encoding_dim):
                reduced_time_series.append(d2_reduced_encoded[i][j])
        list_out.append(reduced_time_series)
        # print("trained: ", len(reduced_time_series))

    else:
        #print(time_series)
        #print(data_to_be_trained)
        tested = X_test[((time_series - data_to_be_trained) * (len(X_test) // (len(df.columns) - data_to_be_trained))):(((time_series - data_to_be_trained) + 1) * (len(X_test) // (len(df.columns) - data_to_be_trained)))]
        reduced_encoded_tested = encoder.predict(tested)
        nsamples, nx, ny = reduced_encoded_tested.shape
        d2_reduced_encoded_tested = reduced_encoded_tested.reshape((nsamples, nx * ny))
        d2_reduced_encoded_tested = sc[time_series].inverse_transform(d2_reduced_encoded_tested)
        #print(d2_reduced_encoded_tested.shape)
        reduced_time_series = []
        for i in range(len(d2_reduced_encoded_tested)):
            for j in range(encoding_dim):
                reduced_time_series.append(d2_reduced_encoded_tested[i][j])
        list_out.append(reduced_time_series)
        # print("tested: ", len(reduced_time_series))

# ------------------------------------------------------------------------------------------------

# ---------------------------------Output Files Computation----------------------------------------
out_f = open('out.txt', 'w')

print(len(list_out))

for i in range(len(list_out) - 10):
    row = list_names[i] + '\t'
    listToStr = '\t'.join([str(elem) for elem in list_out[i]])
    row = row + listToStr
    out_f.write(row)
    out_f.write('\n')

out_f.close()

out = pd.read_csv(r'out.txt')
out.to_csv(r'out.csv', index = None)
os.rename('out.csv', output_file)
os.remove('out.txt')

q_f = open('q.txt', 'w')

for i in range((len(list_out)- 10), len(list_out)):
    row = list_names[i] + '\t'
    listToStr = '\t'.join([str(elem) for elem in list_out[i]])
    row = row + listToStr
    q_f.write(row)
    q_f.write('\n')

q_f.close()

q = pd.read_csv(r'q.txt')
q.to_csv(r'q.csv', index = None)
os.rename('q.csv', output_q)
os.remove('q.txt')
