## Model: Informer

L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980

Predict batch size: 256

2026-02-26 17:40:13.434954: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Trainable parameters: 75521
2026-02-26 17:40:18.179512: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
2026-02-26 17:40:24.351286: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2026-02-26 17:40:24.421379: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8907
2134/2134 [\=\=\=\=\=\=] - 54s 23ms/step - loss: 0.0344 - val_loss: 4.7827e-04
Epoch 2/10
2134/2134 [\=\=\=\=\=\=] - 43s 20ms/step - loss: 0.0013 - val_loss: 8.2277e-05
Epoch 3/10
2134/2134 [\=\=\=\=\=\=] - 47s 22ms/step - loss: 3.7437e-04 - val_loss: 5.9129e-05
Epoch 4/10
2134/2134 [\=\=\=\=\=\=] - 44s 20ms/step - loss: 3.5190e-04 - val_loss: 2.2573e-05
Epoch 5/10
2134/2134 [\=\=\=\=\=\=] - 44s 21ms/step - loss: 1.9389e-04 - val_loss: 1.3522e-05
Epoch 6/10
2134/2134 [\=\=\=\=\=\=] - 47s 22ms/step - loss: 1.7002e-04 - val_loss: 3.4671e-05
Epoch 7/10
2134/2134 [\=\=\=\=\=\=] - 48s 22ms/step - loss: 1.4102e-04 - val_loss: 4.0630e-05
Epoch 8/10
2134/2134 [\=\=\=\=\=\=] - 44s 21ms/step - loss: 1.4001e-04 - val_loss: 8.3995e-06
Epoch 9/10
2134/2134 [\=\=\=\=\=\=] - 43s 20ms/step - loss: 9.7182e-05 - val_loss: 4.3457e-05
Epoch 10/10
2134/2134 [\=\=\=\=\=\=] - 44s 20ms/step - loss: 6.9116e-04 - val_loss: 2.3595e-05
2026-02-26 17:47:57.850881: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.
2026-02-26 17:47:58.345293: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.

Test metrics (original scale):
  Step-1 RMSE:  1.3518
  Step-1 MAPE:  0.0613%
  Horizon RMSE: 1.0744
  Horizon MAPE: 0.0467%

Saved predictions: outputs\predictions_informer_L168_H24.csv
Saved model:       outputs\model_informer_L168_H24
Saved plot:        outputs\actual_vs_pred_informer.png