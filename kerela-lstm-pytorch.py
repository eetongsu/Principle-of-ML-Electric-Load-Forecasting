#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['year'] = df.index.year
    df['season'] = df['month'] % 12 // 3 + 1
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Additional features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['dayofmonth'] == 1).astype(int)
    df['is_month_end'] = (df['dayofmonth'] == df.index.days_in_month).astype(int)
    df['is_quarter_start'] = ((df['dayofmonth'] == 1) & (df['month'] % 3 == 1)).astype(int)
    df['is_quarter_end'] = (df['dayofmonth'] == df.groupby(['year', 'quarter'])['dayofmonth'].transform('max')).astype(int)


    # Additional features
    df['is_working_day'] = df['dayofweek'].isin([0, 1, 2, 3, 4]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 12, 18]).astype(int)

    # Minute-level features
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['minute_of_week'] = (df['dayofweek'] * 24 * 60) + df['minute_of_day']

    return df


data = pd.read_csv("load_forecasting_dataset_corrected.csv")

# Create features for the new data
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data = data.set_index('Timestamp')

data = create_features(data)

# Convert 'Season' to numerical using one-hot encoding
if 'Season' in data.columns:
    data = pd.get_dummies(data, columns=['Season'], prefix='season', dtype=int)
else:
    print("Warning: 'Season' column not found in the DataFrame.")

# Convert remaining columns to float
data = data.astype(float)

# Separate features (X_new) and target (y_new)
X = data.drop('Load Demand (kW)', axis=1)
y = data[['Load Demand (kW)']]

# Split the data into training and testing sets
# Keeping test_size consistent with previous split (96)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=False)

# Initialize StandardScaler for X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fit and transform X_train and y_train
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Transform X_test and y_test using the scalers fitted on the training data
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

print("Shape of scaled training features:", X_train_scaled.shape)
print("Shape of scaled training target:", y_train_scaled.shape)
print("Shape of scaled testing features:", X_test_scaled.shape)
print("Shape of scaled testing target:", y_test_scaled.shape)


# Reshape the input data for LSTM (samples, timesteps, features)
# The LSTM expects input in the shape (batch_size, seq_length, input_size)
# Since we are using a sequence length of 1, we reshape to (samples, 1, features)
X_train_scaled_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_scaled_reshaped).float()
y_train_tensor = torch.from_numpy(y_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled_reshaped).float()
y_test_tensor = torch.from_numpy(y_test_scaled).float()


# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 256 # Using the previously defined batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Number of batches in training loader:", len(train_loader))
print("Number of batches in testing loader:", len(test_loader))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])
        return out

# Instantiate the model
input_size = X_train_tensor.shape[2] # Number of features
hidden_size = 50 # Same as the previous Keras model
num_layers = 1 # Keeping it simple for now

model = LSTMModel(input_size, hidden_size, num_layers)

# Print the model architecture
print(model)

# Define the loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Define the optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Loss function defined:", criterion)
print("Optimizer defined:", optimizer)


# ## Train the pytorch lstm model
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Move model to device
model.to(device)

epochs = 20 # Using the previously defined epochs
# Training loop
for epoch in range(epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

    # Optional: Evaluate on validation set
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(test_loader.dataset)
    print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

print('Finished Training')

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    test_predictions = []
    for inputs, targets in test_loader:
        # Move data to device
        inputs = inputs.to(device)

        # Get model predictions
        outputs = model(inputs)

        # Append predictions to the list
        test_predictions.append(outputs.cpu().numpy())

# Concatenate predictions
test_predictions = np.concatenate(test_predictions, axis=0)

# Convert actual test targets to NumPy array (if not already)
y_test_actual = y_test_tensor.cpu().numpy()

# # Calculate MSE and MAE

# Inverse transform the scaled predictions and actual values
test_predictions_inv = scaler_y.inverse_transform(test_predictions)
y_test_actual_inv = scaler_y.inverse_transform(y_test_actual)

# Create a Matplotlib figure and axes
plt.figure(figsize=(12, 6))

# Plot actual vs predicted for the test set
# Use the index from the original data for the test set
plt.plot(X_test.index, y_test_actual_inv, label='Actual')
plt.plot(X_test.index, test_predictions_inv, label='Predicted')

# Add title and labels
plt.title('Actual vs Predicted Load Demand (PyTorch LSTM)')
plt.xlabel('Timestamp')
plt.ylabel('Load Demand (kW)')

# Add legend
plt.legend()

# Display the plot
plt.show()

# Inverse transform scaled predictions and actual values to calculate metrics on original scale
test_predictions_inv = scaler_y.inverse_transform(test_predictions)
y_test_actual_inv = scaler_y.inverse_transform(y_test_actual)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual_inv, test_predictions_inv))

# Calculate MAPE
# Avoid division by zero in MAPE calculation
mape = np.mean(np.abs((y_test_actual_inv - test_predictions_inv) / y_test_actual_inv)) * 100


# Print the results
print("Root Mean Squared Error (RMSE) on test set: {:.4f}".format(rmse))
print("Mean Absolute Percentage Error (MAPE) on test set: {:.4f}%".format(mape))
