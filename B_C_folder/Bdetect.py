import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import StandardScaler

# load model training or run model training
loadModel = True

# 1 time series default
num = 1
mae_val = 0.0
TIME_STEPS = 30

# ----------------------INPUT PARSER for python --------------------------------------------------
# by https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
# Required dataset argument
parser.add_argument('-d', help='A required string positional argument')
# Optional argument
parser.add_argument('-n', type=int, help='An optional integer argument')
# Optional argument
parser.add_argument('-mae', type=float, help='An optional integer argument')
args = parser.parse_args()
if args.n != None:
    num = args.n
dataset = args.d
if args.d == None:
    dataset = input("Enter a dataset filepath: ")
if args.mae != None:
    mae_val = args.mae
else:
    mae_val = float(input("Enter a mae value: "))
# ------------------------------------------------------------------------------------------------

df = pd.read_csv(dataset, sep = '\t')

# get number of columns and add column keys
col_names = []
col_size = len(df.columns) - 1
for n in range(col_size):
    col_names.append(str(n))

# read csv file again and transpose it
df = pd.read_csv(dataset, sep = '\t', names = col_names)
df = pd.DataFrame.transpose(df)

# keep time series names for plots later
list_names = list(df.columns)

# train almost 80% of data
data_to_be_trained = (80 * len(df.columns)) // 100

scaler_list = []
for i in range((data_to_be_trained + num)):
    scaler_list.append(StandardScaler())
scaler = np.array(scaler_list)

X_train, y_train = [], []
X_test, y_test = [], []
test_list = []
for time_series in range(data_to_be_trained):
    training_set = df.iloc[:, time_series:(time_series+1)].values

    # scaler = StandardScaler()
    # scaler = scaler.fit(training_set)

    training_set_scaled = scaler[time_series].fit_transform(training_set)

    # reshape to [samples, time_steps, n_features]
    for i in range(len(training_set_scaled) - TIME_STEPS):
        v = training_set_scaled[i:(i + TIME_STEPS)]
        X_train.append(v)
        y_train.append(training_set_scaled[i + TIME_STEPS])

X_train, y_train = np.array(X_train), np.array(y_train)

for time_series in range(data_to_be_trained, (data_to_be_trained + num)):
    test_set = df.iloc[:, time_series:(time_series+1)].values

    # scaler = StandardScaler()
    # scaler = scaler.fit(test_set)
    test_set_scaled = scaler[time_series].fit_transform(test_set)
    test_list.append(test_set_scaled)

    for i in range(len(test_set_scaled) - TIME_STEPS):
        v = test_set_scaled[i:(i + TIME_STEPS)]
        X_test.append(v)
        y_test.append(test_set_scaled[i + TIME_STEPS])


X_test, y_test = np.array(X_test), np.array(y_test)

print(X_train.shape)

if loadModel is True:
    model = tf.keras.models.load_model('saved_models/model_detect')

    # Check its architecture
    model.summary()

    # Evaluate the restored model
    # loss, acc = model.evaluate(X_train, y_train, verbose=2)
    # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
else :
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
    model.add(keras.layers.LSTM(units=64, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
    model.compile(loss='mae', optimizer='adam')

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, shuffle=False)

    # Save the entire model as a SavedModel.
    model.save('saved_models/model_detect')

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

THRESHOLD = mae_val
for time_series in range(num):
    # train = X_train[(time_series*(len(X_train) // num)):((time_series+1)*(len(X_train) // num))]
    # X_train_pred = model.predict(train)
    # train_mae_loss = np.mean(np.abs(X_train_pred - train), axis=1)
    #
    # sns.distplot(train_mae_loss, bins=50, kde=True)
    # plt.show()

    test_x = X_test[(time_series*(len(X_test) // num)):((time_series+1)*(len(X_test) // num))]
    # print(test_x.shape)
    X_test_pred = model.predict(test_x)
    test_mae_loss = np.mean(np.abs(X_test_pred - test_x), axis=1)

    test_score_df = pd.DataFrame(index=col_names[TIME_STEPS:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
    test_score_df['value'] = test_list[time_series][TIME_STEPS:]

    plt.plot(test_score_df.index, test_score_df.loss, label='loss')
    plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
    plt.xticks(np.arange(0,730,30))
    plt.legend()
    plt.show()

    anomalies = test_score_df[test_score_df.anomaly == True]
    print(anomalies.head())

    # crate title for plot
    title = 'Time Series '
    for ts_name in list_names[(data_to_be_trained + time_series):((data_to_be_trained + time_series) + 1)]:
        title = title + ts_name
    title = title + ' Anomaly Detection'

    # inverse transform
    plt.plot(col_names[TIME_STEPS:], test_list[time_series][TIME_STEPS:], label='time series value')
    sns.scatterplot(anomalies.index, anomalies.value, color=sns.color_palette()[3], s=52, label='anomaly')
    plt.xticks(np.arange(0,730,30))
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
