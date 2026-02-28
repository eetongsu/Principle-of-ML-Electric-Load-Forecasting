## Results

Model: rnn
L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980

Predict batch size: 256
=============================
2026-02-26 23:13:35.729154: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
WARNING:tensorflow:Layer gru will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
Trainable parameters: 47320
2026-02-26 23:13:38.977258: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
2026-02-26 23:13:44.621093: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2134/2134 [==============================] - 1140s 532ms/step - loss: 1.0006 - val_loss: 0.9812
Epoch 2/10
2134/2134 [==============================] - 1119s 524ms/step - loss: 1.0004 - val_loss: 0.9811
Epoch 3/10
2134/2134 [==============================] - 1123s 526ms/step - loss: 1.0003 - val_loss: 0.9813
Epoch 4/10
2134/2134 [==============================] - 1129s 529ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 5/10
2134/2134 [==============================] - 1139s 534ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 6/10
2134/2134 [==============================] - 1142s 535ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 7/10
2134/2134 [==============================] - 1141s 535ms/step - loss: 1.0003 - val_loss: 0.9812
Epoch 8/10
2134/2134 [==============================] - 37695s 18s/step - loss: 1.0003 - val_loss: 0.9812
Epoch 9/10
2134/2134 [==============================] - 1137s 533ms/step - loss: 1.0003 - val_loss: 0.9813
Epoch 10/10
2134/2134 [==============================] - 1136s 532ms/step - loss: 1.0003 - val_loss: 0.9811
2026-02-27 12:32:03.651259: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.
2026-02-27 12:32:04.095590: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.

Test metrics (original scale):
  Step-1 RMSE:  199.4769
  Step-1 MAPE:  11.0203%
  Horizon RMSE: 199.4952
  Horizon MAPE: 11.0264%

Saved predictions: outputs\predictions_rnn_L168_H24.csv
Saved model:       outputs\model_rnn_L168_H24
Saved keras model: outputs\keras_savedmodel_rnn_L168_H24
Saved plot:        outputs\actual_vs_pred_rnn.png