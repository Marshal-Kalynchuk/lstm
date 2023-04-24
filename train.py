
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

import seaborn as sns

tf.keras.backend.clear_session()


# Load the CSV data into a pandas DataFrame
df = pd.read_csv('SOL-USD.csv')
train_dates = pd.to_datetime(df['Date'])

cols = list(df)[1:7]
df_for_training = df[cols].astype(float)

df_for_plot=df_for_training.tail(100)
df_for_plot.plot.line()

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


x_train = []
y_train = []

n_future = 7
n_past = 60

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    x_train.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    y_train.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train.shape)
print(y_train.shape)

#x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units = 50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, activation='relu', return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, activation='relu'))
regressor.add(Dropout(0.2))

regressor.add(Dense(y_train.shape[1]))

regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')


history = regressor.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.1, verbose=1)

n_future = 90
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = regressor.predict(x_train[-n_future:])

forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})

df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])

# filter dates before '2023-01-20'
original = df.loc[df['Date'] >= '2022-09-20', ['Date', 'Open']]

# convert 'Date' column to datetime
original['Date'] = pd.to_datetime(original['Date'])

print(original.head())
print(df_forecast.head())

# plot the data
plt.figure(figsize=(10, 5))
sns.lineplot(x='Date', y='Open', data=original)
sns.lineplot(x='Date', y='Open', data=df_forecast)
plt.xlabel('Date')
plt.ylabel('Open')
plt.show()