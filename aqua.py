import gc
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.layers import ReLU
from keras.layers import PReLU
import seaborn as sns
import ta.momentum
import ta.volume
import ta.volatility
import ta.trend
from sklearn.model_selection import train_test_split
import mdn

from keras.callbacks import ReduceLROnPlateau

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

mixed_precision = 'mixed_float16'
tf.keras.mixed_precision.set_global_policy(mixed_precision)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


N_PAST = 1 # 2 hours
N_FUTURE = 1 # 5 minutes
N_TEST = 6000
N_VALIDATION = 20000
BATCH_SIZE = 1
DATASET_SIZE = 140000
EPOCHS = 1
STEPS_PER_EPOCH = ((DATASET_SIZE - N_TEST - N_VALIDATION - N_PAST - N_FUTURE)//BATCH_SIZE)//8


TARGET_COLUMN = 'target'
#INPUT_COLUMNS = ['rsi', 'stochastic', 'open', 'close', 'high', 'low', 'volume']
INPUT_COLUMNS = ['open', 'close', 'high', 'low', 'volume', 'stochastic_xshort', 'stochastic_short', 'stochastic_med', 'stochastic_long', 'stochastic_xlong', 'rsi_xshort', 'rsi_short', 'rsi_med','rsi_long', 'rsi_xlong', 'kama', 'roc_xshort', 'roc_short', 'roc_med', 'roc_long',
                 'roc_xlong', 'tsi', 'william', 'pvo', 'ult_oscillator', 'ppo', 'cmf', 'eom', 'fi_xshort', 'fi_short', 'fi_med', 'fi_long', 'fi_xlong', 'mfi', 'atr_xshort', 'atr_short', 'atr_med', 'atr_long', 
                 'bollinger_pband', 'donchian_pband', 'ui','aroon_up', 'aroon_down', 'cci', 'dpo', 'kst', 'mass', 'trix_xshort', 'trix_short', 'trix_med', 'trix_long', 'trix_xlong', 'vortex_pos', 'vortex_neg', 'macd_xshort', 'macd_short', 'macd_med', 'macd_long', 'macd_diff',
                 'ema_xshort', 'ema_short', 'ema_med', 'ema_long', 'ema_xlong',]

N_PARAMS = len(INPUT_COLUMNS)
print(N_PARAMS)

tf.keras.backend.clear_session()

df = pd.read_csv('solusd.csv')
print(df.head(10))

df['time'] = df['time'].astype(float).astype(int)
df['time'] = pd.to_datetime(df['time'], unit='ms')

df = df.groupby(pd.Grouper(key='time', freq='5min')).agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).reset_index()

print(df.head(10))

# df = df.drop(index=df.index[:len(df)//2])
df[['open', 'close', 'high', 'low', 'volume']] = df[['open', 'close', 'high', 'low', 'volume']].astype(float)

# Create a target of relative difference
#df['target'] = (df['close'] - df['close'].shift(N_FUTURE))/df['close']

# log return
df['target'] = np.log(df['close']) - np.log(df['close'].shift(N_FUTURE))

# Momentum
df['stochastic_xshort'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=3)
df['stochastic_short'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=5)
df['stochastic_med'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=7)
df['stochastic_long'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
df['stochastic_xlong'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=32)
df['rsi_xshort'] = ta.momentum.rsi(df['close'], window=3)
df['rsi_short'] = ta.momentum.rsi(df['close'], window=5)
df['rsi_med'] = ta.momentum.rsi(df['close'], window=7)
df['rsi_long'] = ta.momentum.rsi(df['close'], window=14)
df['rsi_xlong'] = ta.momentum.rsi(df['close'], window=32)
df['kama'] = ta.momentum.kama(df['close'])
df['roc_xshort'] = ta.momentum.roc(df['close'], window=3)
df['roc_short'] = ta.momentum.roc(df['close'], window=5)
df['roc_med'] = ta.momentum.roc(df['close'], window=7)
df['roc_long'] = ta.momentum.roc(df['close'], window=14)
df['roc_xlong'] = ta.momentum.roc(df['close'], window=32)
df['tsi'] = ta.momentum.tsi(df['close'])
df['william'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
df['pvo'] = ta.momentum.pvo(df['volume'])
df['ult_oscillator'] = ta.momentum.ultimate_oscillator(df['high'], df['low'], df['close'])
df['ppo'] = ta.momentum.ppo(df['close'])

# Volume
df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'])
df['fi_xshort'] = ta.volume.force_index(df['close'], df['volume'], window=3)
df['fi_short'] = ta.volume.force_index(df['close'], df['volume'], window=5)
df['fi_med'] = ta.volume.force_index(df['close'], df['volume'], window=7)
df['fi_long'] = ta.volume.force_index(df['close'], df['volume'], window=14)
df['fi_xlong'] = ta.volume.force_index(df['close'], df['volume'], window=32)
df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])

# Volatility
df['atr_xshort'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=3)
df['atr_short'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=5)
df['atr_med'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=7)
df['atr_long'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
df['bollinger_pband'] = ta.volatility.bollinger_pband(df['close'])
df['donchian_pband'] = ta.volatility.donchian_channel_pband(df['high'], df['low'], df['close'])
# df['keltner_pband'] = ta.volatility.keltner_channel_pband(df['high'], df['low'], df['close']) # breaks the progrem
df['ui'] = ta.volatility.ulcer_index(df['close'])

# Trend
df['aroon_up'] = ta.trend.aroon_up(df['close'])
df['aroon_down'] = ta.trend.aroon_down(df['close'])
df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
df['dpo'] = ta.trend.dpo(df['close'])
df['kst'] = ta.trend.kst(df['close'])
df['mass'] = ta.trend.mass_index(df['high'], df['low'])
df['trix_xshort'] = ta.trend.trix(df['close'], window=3)
df['trix_short'] = ta.trend.trix(df['close'], window=5)
df['trix_med'] = ta.trend.trix(df['close'], window=7)
df['trix_long'] = ta.trend.trix(df['close'], window=14)
df['trix_xlong'] = ta.trend.trix(df['close'], window=32)
df['vortex_pos'] = ta.trend.vortex_indicator_pos(df['high'], df['low'], df['close'])
df['vortex_neg'] = ta.trend.vortex_indicator_neg(df['high'], df['low'], df['close'])

df['macd_xshort'] = ta.trend.macd(df['close'], 6, 2)
df['macd_short'] = ta.trend.macd(df['close'], 12, 6)
df['macd_med'] = ta.trend.macd(df['close'], 24, 12)
df['macd_long'] = ta.trend.macd(df['close'], 52, 26)
df['macd_diff'] = ta.trend.macd(df['close'], 52, 6)

df['ema_xshort'] = ta.trend.ema_indicator(df['close'], window=3)
df['ema_short'] = ta.trend.ema_indicator(df['close'], window=5)
df['ema_med'] = ta.trend.ema_indicator(df['close'], window=7)
df['ema_long'] = ta.trend.ema_indicator(df['close'], window=14)
df['ema_xlong'] = ta.trend.ema_indicator(df['close'], window=32)


df.fillna(0, inplace=True)
print(df.head(10))
print(df.tail(10))

train_data = df[:-N_TEST]
test_data = df[-N_TEST:]

print(train_data.head(10))
# Prepare training data
train_targets = train_data[[TARGET_COLUMN]].astype(np.float32)
train_variables = train_data[INPUT_COLUMNS].astype(np.float32)

print("\n\nTrain Targets Shape:")
print(train_targets.shape)
print("\nTrain Targets Tail:")
print(train_targets.tail(10))

print("\n\nTrain Variables Shape:")
print(train_variables.shape)
print("\nTrain Variables Tail:")
print( train_variables.tail(10))

# create scalers
target_scaler = StandardScaler()
target_scaler = target_scaler.fit(train_targets)
scaled_train_targets = target_scaler.transform(train_targets)

del train_targets

variables_scaler = StandardScaler()
variables_scaler = variables_scaler.fit(train_variables)
scaled_train_variables = variables_scaler.transform(train_variables)

del train_variables

# scale traininng data
scaled_train_variables_df = pd.DataFrame(scaled_train_variables, index=train_data.index, columns=INPUT_COLUMNS)
scaled_train_targets_df = pd.DataFrame(scaled_train_targets, index=train_data.index, columns=[TARGET_COLUMN])


train_dataset = tf.data.Dataset.from_tensor_slices(scaled_train_targets)
train_dataset.batch()

scaled_train_data_df = pd.concat([scaled_train_targets_df, scaled_train_variables_df], axis=1).fillna(value=0)

del scaled_train_variables_df
del scaled_train_targets_df

# print("\n\nScaled training data")
# print(scaled_train_data_df.shape)
# print(scaled_train_data_df.head(100))
# print(scaled_train_data_df.tail(100))


# prepare testing data
test_targets = test_data[[TARGET_COLUMN]].astype(float)
test_variables = test_data[INPUT_COLUMNS].astype(float)

# print("\n\nTest Targets Shape:")
# print(test_targets.shape)
# print("\nTest Targets Tail:")
# print(test_targets.tail(10))
# 
# print("\n\nTest Variables Shape:")
# print(test_variables.shape)
# print("\nTest Variables Tail:")
# print(test_variables.tail(10))

scaled_test_targets = target_scaler.transform(test_targets)
scaled_test_variables = variables_scaler.transform(test_variables)

del test_targets
del test_variables

scaled_test_variables_df = pd.DataFrame(scaled_test_variables, index=test_data.index, columns=INPUT_COLUMNS)
scaled_test_targets_df = pd.DataFrame(scaled_test_targets, index=test_data.index, columns=[TARGET_COLUMN])
scaled_test_data_df = pd.concat([scaled_test_targets_df, scaled_test_variables_df], axis=1).fillna(value=0)

del scaled_test_variables_df
del scaled_test_targets_df

del train_data
del test_data

# print("\n\nScaled testing data")
# print(scaled_test_data_df.shape)
# print(scaled_test_data_df.head())

def create_rolling_window(data_df, n_past, n_future):
    x_data = []
    y_data = []
    for i in range(n_past, len(data_df) - n_future + 1):

        x_data.append(data_df[INPUT_COLUMNS][i - n_past:i])
        y_data.append(data_df[TARGET_COLUMN][i + n_future - 1: i + n_future])

    return np.array(x_data), np.array(y_data)


def data_generator(data_df, n_past, n_future, batch_size):
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            # select a random index to start the sequence
            idx = np.random.randint(n_past, len(data_df) - n_future + 1)
            # extract the sequence and target data
            seq = data_df[INPUT_COLUMNS][idx - n_past:idx]
            target = data_df[[TARGET_COLUMN]][idx + n_future - 1: idx + n_future]
            
            x_batch.append(seq)
            y_batch.append(target)

        yield np.array(x_batch), np.array(y_batch)


def create_lstm():

    regressor = Sequential()

    regressor.add(CuDNNLSTM(units = 30, return_sequences=True, input_shape=(N_PAST, N_PARAMS)))
    regressor.add(Dropout(0.3))

    regressor.add(CuDNNLSTM(units = 30, return_sequences=True))
    regressor.add(Dropout(0.3))

    regressor.add(CuDNNLSTM(units = 30))
    regressor.add(Dropout(0.3))

    #regressor.add(mdn.MDN(1, 5))

    regressor.add(ReLU())
    regressor.add(Dense(1))

    regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')

    return regressor

reduce_lr = ReduceLROnPlateau(monitor='val_loss')

train_df = scaled_train_data_df[:-N_VALIDATION]
val_df = scaled_train_data_df[-N_VALIDATION:]

train_gen =  data_generator(train_df, N_PAST, N_FUTURE, BATCH_SIZE)
val_gen = data_generator(val_df, N_PAST, N_FUTURE, BATCH_SIZE)

tf.keras.backend.clear_session()

regressor = create_lstm()

history = regressor.fit(train_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1, callbacks=[reduce_lr], validation_data=val_gen, validation_steps=BATCH_SIZE)

test_data_gen = data_generator(scaled_test_data_df, N_PAST, N_FUTURE, BATCH_SIZE)
mse = regressor.evaluate(test_data_gen, steps=STEPS_PER_EPOCH)
print("MSE on test set: {:.10f}".format(mse))

test_x, test_y = create_rolling_window(scaled_test_data_df.tail(1000), N_PAST, N_FUTURE)
print(test_x.shape)
pred_y_scaled = regressor.predict(test_x)

test_y = target_scaler.inverse_transform(test_y.reshape(1, -1)).flatten()

pred_y = target_scaler.inverse_transform(pred_y_scaled).flatten()


sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(x=range(len(test_y)), y=test_y, ax=ax, label="Actual")
sns.lineplot(x=range(len(pred_y)), y=pred_y, ax=ax, label="Predicted")

ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
