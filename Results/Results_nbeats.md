## Model: N-BEATS

L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980

Predict batch size: 256

2026-02-28 13:31:58.018594: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
NBeats only support single variable prediction, so ignore encoder_features and decoder_features
Trainable parameters: 150144
2026-02-28 13:32:00.925437: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
2026-02-28 13:32:06.679475: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2134/2134 [==============================] - 34s 14ms/step - loss: 1.0098 - val_loss: 0.9810
Epoch 2/10
2134/2134 [==============================] - 28s 13ms/step - loss: 1.0002 - val_loss: 0.9810
Epoch 3/10
2134/2134 [==============================] - 29s 13ms/step - loss: 1.0002 - val_loss: 0.9810
Epoch 4/10
2134/2134 [==============================] - 28s 13ms/step - loss: 1.0002 - val_loss: 0.9810
Epoch 5/10
2134/2134 [==============================] - 29s 13ms/step - loss: 1.0001 - val_loss: 0.9810
Epoch 6/10
2134/2134 [==============================] - 30s 14ms/step - loss: 1.0001 - val_loss: 0.9810
Epoch 7/10
2134/2134 [==============================] - 30s 14ms/step - loss: 1.0001 - val_loss: 0.9810
Epoch 8/10
2134/2134 [==============================] - 31s 14ms/step - loss: 1.0001 - val_loss: 0.9810
Epoch 9/10
2134/2134 [==============================] - 29s 14ms/step - loss: 1.0001 - val_loss: 0.9810
Epoch 10/10
2134/2134 [==============================] - 30s 14ms/step - loss: 1.0001 - val_loss: 0.9810

Test metrics (original scale):
  Step-1 RMSE:  199.4821
  Step-1 MAPE:  11.0228%
  Horizon RMSE: 199.4800
  Horizon MAPE: 11.0230%

Saved predictions: outputs\predictions_nbeats_L168_H24.csv
Saved model:       outputs\model_nbeats_L168_H24
WARNING:absl:Found untraced functions such as dense_4_layer_call_fn, dense_4_layer_call_and_return_conditional_losses, dense_4_layer_call_fn, dense_4_layer_call_and_return_conditional_losses, dense_4_layer_call_fn while saving (showing 5 of 60). These functions will not be directly callable after loading.
Saved keras model: outputs\keras_savedmodel_nbeats_L168_H24
Saved plot:        outputs\actual_vs_pred_nbeats.png