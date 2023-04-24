import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = torch.load('LSTM.pt')

# Load and preprocess new data
new_data = pd.read_csv('new_data.csv')
scaler = MinMaxScaler()
new_data = scaler.fit_transform(new_data)

# Convert data to PyTorch tensors
new_data_tensor = torch.FloatTensor(new_data).unsqueeze(0)

# Make predictions
predictions = model(new_data_tensor)

# Invert scaling on predictions
predictions = scaler.inverse_transform(predictions.detach().numpy())

# Print predictions
print(predictions)
