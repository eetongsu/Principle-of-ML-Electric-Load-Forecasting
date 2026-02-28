## Model: LSTM

Shape of scaled training features: (37977, 36)
Shape of scaled training target: (37977, 1)
Shape of scaled testing features: (151911, 36)
Shape of scaled testing target: (151911, 1)
Number of batches in training loader: 149
Number of batches in testing loader: 594
LSTMModel(
  (lstm): LSTM(36, 50, batch_first=True)
  (linear): Linear(in_features=50, out_features=1, bias=True)
)
Loss function defined: MSELoss()
Optimizer defined: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
Using device: cuda
Epoch [1/20], Loss: 0.6706
Epoch [1/20], Validation Loss: 0.3670
Epoch [2/20], Loss: 0.0250
Epoch [2/20], Validation Loss: 0.0476
Epoch [3/20], Loss: 0.0047
Epoch [3/20], Validation Loss: 0.0353
Epoch [4/20], Loss: 0.0041
Epoch [4/20], Validation Loss: 0.0312
Epoch [5/20], Loss: 0.0037
Epoch [5/20], Validation Loss: 0.0289
Epoch [6/20], Loss: 0.0033
Epoch [6/20], Validation Loss: 0.0269
Epoch [7/20], Loss: 0.0029
Epoch [7/20], Validation Loss: 0.0226
Epoch [8/20], Loss: 0.0026
Epoch [8/20], Validation Loss: 0.0227
Epoch [9/20], Loss: 0.0023
Epoch [9/20], Validation Loss: 0.0191
Epoch [10/20], Loss: 0.0020
Epoch [10/20], Validation Loss: 0.0182
Epoch [11/20], Loss: 0.0017
Epoch [11/20], Validation Loss: 0.0187
Epoch [12/20], Loss: 0.0014
Epoch [12/20], Validation Loss: 0.0217
Epoch [13/20], Loss: 0.0010
Epoch [13/20], Validation Loss: 0.0209
Epoch [14/20], Loss: 0.0006
Epoch [14/20], Validation Loss: 0.0198
Epoch [15/20], Loss: 0.0004
Epoch [15/20], Validation Loss: 0.0181
Epoch [16/20], Loss: 0.0003
Epoch [16/20], Validation Loss: 0.0162
Epoch [17/20], Loss: 0.0003
Epoch [17/20], Validation Loss: 0.0162
Epoch [18/20], Loss: 0.0002
Epoch [18/20], Validation Loss: 0.0158
Epoch [19/20], Loss: 0.0002
Epoch [19/20], Validation Loss: 0.0150
Epoch [20/20], Loss: 0.0002
Epoch [20/20], Validation Loss: 0.0159
Finished Training
Root Mean Squared Error (RMSE) on test set: 25.2153
Mean Absolute Percentage Error (MAPE) on test set: 1.1892%

Process finished with exit code 0