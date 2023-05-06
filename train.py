import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import CuDNNLSTM
from keras.layers import Dense
from keras.layers import Dropout
import seaborn as sns

N_PAST = 740 # 0.5 days
N_FUTURE = 15 # 15minutes
N_TEST = 5000
BATCH_SIZE = 256
DATASET_SIZE = 700000
EPOCHS = 10
STEPS_PER_EPOCH = 200 # (DATASET_SIZE - N_TEST)//BATCH_SIZE
N_PARAMS = 5

tf.keras.backend.clear_session()

df = pd.read_csv('solusd.csv')

train_data = df[:-N_TEST]
test_data = df[-N_TEST:]

# get dates
dates = pd.to_datetime(df['time'], unit='ms')


# Prepare training data
train_targets = list(train_data)[1:2]
train_variables = list(train_data)[2:6]

train_targets = train_data[train_targets].astype(float)
train_variables = train_data[train_variables].astype(float)

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

variables_scaler = StandardScaler()
variables_scaler = variables_scaler.fit(train_variables)
scaled_train_variables = variables_scaler.transform(train_variables)

# scale traininng data
scaled_train_variables_df = pd.DataFrame(scaled_train_variables, index=train_data.index, columns=train_data.columns[2:6])
scaled_train_targets_df = pd.DataFrame(scaled_train_targets, index=train_data.index, columns=[train_data.columns[1]])
scaled_train_data_df = pd.concat([scaled_train_targets_df, scaled_train_variables_df], axis=1)

print("\n\nScaled training data")
print(scaled_train_data_df.shape)
print(scaled_train_data_df.head())


# prepare testing data
test_targets = list(test_data)[1:2]
test_variables = list(test_data)[2:6]

test_targets = test_data[test_targets].astype(float)
test_variables = test_data[test_variables].astype(float)

print("\n\nTest Targets Shape:")
print(test_targets.shape)
print("\nTest Targets Tail:")
print(test_targets.tail(10))

print("\n\nTest Variables Shape:")
print(test_variables.shape)
print("\nTest Variables Tail:")
print(test_variables.tail(10))

scaled_test_targets = target_scaler.transform(test_targets)
scaled_test_variables = variables_scaler.transform(test_variables)

scaled_test_variables_df = pd.DataFrame(scaled_test_variables, index=test_data.index, columns=test_data.columns[2:6])
scaled_test_targets_df = pd.DataFrame(scaled_test_targets, index=test_data.index, columns=[test_data.columns[1]])
scaled_test_data_df = pd.concat([scaled_test_targets_df, scaled_test_variables_df], axis=1)

print("\n\nScaled testing data")
print(scaled_test_data_df.shape)
print(scaled_test_data_df.head())

def create_rolling_window(data_tf, n_past, n_future):
    x_data = []
    y_data = []
    for i in range(n_past, len(data_tf) - n_future + 1):
        x_data.append(data_tf[i - n_past:i])
        y_data.append(data_tf[i + n_future - 1: i + n_future]['open'])

    return np.array(x_data), np.array(y_data)


def data_generator(data_df, n_past, n_future, batch_size):
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            # select a random index to start the sequence
            idx = np.random.randint(n_past, len(data_df) - n_future + 1)
            # extract the sequence and target data
            seq = data_df[idx - n_past:idx]
            target = data_df[idx + n_future - 1:idx + n_future]['open']
            x_batch.append(seq)
            y_batch.append(target)
        yield np.array(x_batch), np.array(y_batch)


def create_lstm():

    regressor = Sequential()

    regressor.add(CuDNNLSTM(units = 50, return_sequences=True, input_shape=(N_PAST, N_PARAMS)))
    regressor.add(Dropout(0.2))

    regressor.add(CuDNNLSTM(units = 50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(CuDNNLSTM(units = 50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(CuDNNLSTM(units = 50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(1))

    regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')

    return regressor


regressor = create_lstm()
train_data_gen = data_generator(scaled_train_data_df, N_PAST, N_FUTURE, BATCH_SIZE)
history = regressor.fit(train_data_gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)

# test_data_gen = data_generator(scaled_test_data_df, N_PAST, N_FUTURE, BATCH_SIZE)
# mse = regressor.evaluate(train_data_gen, steps=len(scaled_test_data_df)//BATCH_SIZE)
# print("MSE on test set: {:.10f}".format(mse))

test_x, test_y = create_rolling_window(scaled_test_data_df, N_PAST, N_FUTURE)
pred_y = regressor.predict(test_x)

predicted = target_scaler.inverse_transform(pred_y).flatten()
actual = target_scaler.inverse_transform(test_y).flatten()

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(x=range(len(actual)), y=actual, ax=ax, label="Actual")
sns.lineplot(x=range(len(predicted)), y=predicted, ax=ax, label="Predicted")
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.legend()
plt.show()
