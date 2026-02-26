#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from PyEMD import EMD  # pip install EMD-signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt


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


# # Feature Engineering and Scaling on New Data
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

emd = EMD()

# Apply EMD to the entire 'Load Demand (kW)' time series
imfs = emd(data['Load Demand (kW)'].values)
# imfs = imfs[:, 0:1000]

# Display the shape of the resulting IMFs
print("Shape of IMFs:", imfs.shape)


# Separate features (X) and target (y) - Keep X_new as is for now, but y_new will be scaled later
X_new = data.drop('Load Demand (kW)', axis=1)
# y_new = data['Load Demand (kW)'].values.reshape(-1, 1) # We will scale y after EMD

# Initialize the scaler for features (X)
scaler_X_new = StandardScaler()

# Scale the features
X_new_scaled = scaler_X_new.fit_transform(X_new)

# We will scale the target variable (Load Demand) after EMD

print("Shape of scaled features:", X_new_scaled.shape)
# print("Shape of scaled target:", y_new_scaled.shape) # This will be printed later


# Define the proportion for the training set
train_proportion = 0.8

# Determine the number of components (IMFs + residual)
num_components = imfs.shape[0]
data_length = imfs.shape[1]

# Calculate the split index
split_index = int(data_length * train_proportion)

# List to store split, scaled, and reshaped data for each component
component_data = []

# Handle potential length differences (assuming they are all the same for this EMD library)
# If different lengths were possible, interpolation or truncation would be needed here.
# For now, we assume all components have the same length as the original data.
min_length = data_length # Assuming all IMFs have the same length as the original data

# Iterate through each component (IMF or residual)
for i in range(num_components):
    # Get the current component and truncate/interpolate if necessary (not needed with this EMD lib)
    component = imfs[i, :min_length]

    # Reshape for scaling (needs to be 2D)
    component_reshaped_for_scaling = component.reshape(-1, 1)

    # Split component into training and testing sets (before scaling)
    component_train_unscaled = component_reshaped_for_scaling[:split_index]
    component_test_unscaled = component_reshaped_for_scaling[split_index:]

    # Initialize and fit a dedicated MinMaxScaler for the current component on the training data only
    scaler = MinMaxScaler()
    component_train_scaled = scaler.fit_transform(component_train_unscaled)

    # Transform the testing data using the scaler fitted on the training data
    component_test_scaled = scaler.transform(component_test_unscaled)


    # Reshape for LSTM input (samples, timesteps, features)
    # Each time step has one feature (the scaled IMF or residual value)
    component_train_reshaped = component_train_scaled.reshape(-1, 1, 1)
    component_test_reshaped = component_test_scaled.reshape(-1, 1, 1)


    # Store based on whether it's an IMF or the residual, and include the scaler
    if i < num_components - 1:
        component_data.append({
            'name': f'imf_{i}',
            'train': component_train_reshaped,
            'test': component_test_reshaped,
            'scaler': scaler # Store the scaler used for this component
        })
    else:
        component_data.append({
            'name': 'residual',
            'train': component_train_reshaped,
            'test': component_test_reshaped,
            'scaler': scaler # Store the scaler used for this component
        })

# Display the shapes of the first IMF and residual as an example
print("Example shapes for IMF 0:")
print("Train shape:", component_data[0]['train'].shape)
print("Test shape:", component_data[0]['test'].shape)

print("\nExample shapes for Residual:")
print("Train shape:", component_data[-1]['train'].shape)
print("Test shape:", component_data[-1]['test'].shape)


# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        # Initialize hidden cell state
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        # Ensure input_seq has shape (seq_len, batch_size, input_size)
        # For our case, seq_len is 1, batch_size is 1, input_size is 1
        # The view(len(input_seq) ,1, -1) reshapes the input for LSTM (seq_len, batch_size, input_size)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        # The view(len(lstm_out), -1) reshapes the output for the linear layer (seq_len, hidden_size)
        predictions = self.linear(lstm_out.view(len(lstm_out), -1))
        return predictions

# Training parameters
epochs = 15 # Set epochs to 30 as per the main task
learning_rate = 0.01 # Set learning rate to 0.01 as per the main task

# Check for GPU availability and set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dictionary to store trained models and their loss history
trained_models = {}
loss_history = {}

# Iterate through each component (IMF or residual)
for component in component_data:
    component_name = component['name']
    print(f"Training model for {component_name}...")

    # Get training and testing data for the current component
    train_data = component['train']
    test_data = component['test'] # This is the validation set

    # Convert data to PyTorch tensors and move to device
    train_tensor = torch.FloatTensor(train_data).to(device)
    test_tensor = torch.FloatTensor(test_data).to(device)

    batch_size = 256

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_tensor, train_tensor) # Labels are the same as input for reconstruction
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(test_tensor, test_tensor) # Labels are the same as input for validation
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # Initialize the model, loss function, and optimizer and move model to device
    model = LSTMModel().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lists to store loss history for the current component
    train_losses = []
    validation_losses = []


    # Training loop with tqdm
    for i in tqdm(range(epochs), desc=f"    Training {component_name}"):
        model.train() # Set model to training mode
        epoch_train_loss = 0

        for seq, labels in train_loader:
            optimizer.zero_grad()
            # Reset hidden cell for each sequence since batch size is 1 and we are not preserving state across sequences
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                torch.zeros(1, 1, model.hidden_layer_size).to(device))

            y_pred = model(seq)

            # Ensure labels have the same shape as predictions for MSELoss
            single_loss = loss_function(y_pred, labels.view_as(y_pred))
            single_loss.backward()
            optimizer.step()
            epoch_train_loss += single_loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        # Calculate validation loss
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            validation_loss = 0
            for seq, labels in test_loader:
                 # Reset hidden cell for each sequence
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                    torch.zeros(1, 1, model.hidden_layer_size).to(device))
                y_pred = model(seq)
                 # Ensure labels have the same shape as predictions for MSELoss
                validation_loss += loss_function(y_pred, labels.view_as(y_pred)).item()
            validation_loss /= len(test_loader)
            validation_losses.append(validation_loss)

        # Print loss every 10 epochs
        if (i+1)%5 == 0:
            print(f'    epoch: {i+1:3} training loss: {train_losses[-1]:10.8f} validation loss: {validation_losses[-1]:10.8f}')

    print(f'    epoch: {epochs:3} last training loss: {train_losses[-1]:10.8f} last validation loss: {validation_losses[-1]:10.8f}')


    # Store the trained model and loss history
    trained_models[component_name] = model
    loss_history[component_name] = {
        'train_loss': train_losses,
        'validation_loss': validation_losses
    }


print("\nAll LSTM models trained.")


# ## Make predictions
# Set the number of prediction points
n_predict = data_length - split_index

predictions = {}

# Iterate through each trained model
for component in component_data:
    component_name = component['name']
    print(f"Generating predictions for {component_name} (last {n_predict} points)...")

    # Check if the model for this component exists
    if component_name in trained_models:
        model = trained_models[component_name]

        # Get the test data for the current component
        test_data = component['test']
        test_tensor = torch.FloatTensor(test_data).to(device) # Move to device

        # Set the model to evaluation mode
        model.eval()

        # Generate predictions
        with torch.no_grad():
            component_predictions = []
            # Reset hidden cell for each sequence
            if hasattr(model, 'hidden_cell'): # Check if the model has a hidden_cell attribute
                 model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                     torch.zeros(1, 1, model.hidden_layer_size).to(device))

            for seq in test_tensor:
                 # Ensure input sequence has the correct shape (1, 1, 1) for batch_size=1, timesteps=1, input_size=1
                seq = seq.view(1, 1, 1)
                y_pred = model(seq)
                component_predictions.append(y_pred.item())

        # Convert predictions to a numpy array and store the last n_predict points
        predictions[component_name] = np.array(component_predictions[-n_predict:])
    else:
        print(f"  Skipping predictions for {component_name}: Trained model not found.")


print("\nAll component predictions generated.")

reconstructed_predictions = None
print(f"Reconstructing overall predictions (last {n_predict} points)...")

# Iterate through each component's predictions
for component_name, component_predictions in predictions.items():
    if reconstructed_predictions is None:
        reconstructed_predictions = component_predictions
    else:
        reconstructed_predictions += component_predictions

if reconstructed_predictions is not None:
    print(f"  Overall reconstructed predictions generated with shape {reconstructed_predictions.shape}.")
else:
    print("  No predictions to reconstruct.")


print("\nOverall reconstructed predictions generated.")


# Get the actual values for the last n_predict points of the test set
# This assumes 'data' DataFrame and 'split_index' are available from previous steps.
# Also assumes 'Load Demand (kW)' is the target column.
actual_values_last_n = data['Load Demand (kW)'].values[split_index:][-n_predict:]


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values_last_n, reconstructed_predictions))

# Calculate MAPE
mape = mean_absolute_percentage_error(actual_values_last_n, reconstructed_predictions) * 100 # Convert to percentage

print(f"Root Mean Squared Error (RMSE) for the last {n_predict} points: {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE) for the last {n_predict} points: {mape:.4f}%")


# ## Data Visualization
# Plot actual vs predicted load demand for the last 96 points
plt.figure(figsize=(12, 6))
# Get the index for the last n_predict points of the test set
actual_index_last_n = data.index[split_index:][-n_predict:]
plt.plot(actual_index_last_n, actual_values_last_n, label='Actual Load Demand', marker='o', linestyle='-')
plt.plot(actual_index_last_n, reconstructed_predictions, label='Predicted Load Demand', marker='x', linestyle='--')
plt.title(f'Actual vs Predicted Load Demand (Last {n_predict} Points)')
plt.xlabel('Timestamp')
plt.ylabel('Load Demand (kW)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training and validation loss for each LSTM model
# This assumes 'loss_history' dictionary contains the loss history.
# This is a placeholder as training was skipped.
# In a real run, you would have this populated from the training cell.
if 'loss_history' in globals() and loss_history:
    print("\nPlotting Training and Validation Loss History:")
    for component_name, history in loss_history.items():
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['validation_loss'], label='Validation Loss')
        plt.title(f'{component_name} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
else:
    print("\nLoss history not available for plotting (Training was skipped or loss history not stored).")


print("Task Complete: EMD+LSTM Model Evaluation Summary\n")

print(f"Evaluation Metrics for the last {n_predict} points:")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f}%")




