Model: unet
L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980
Predict batch size: 256
=============================
2026-02-27 16:14:34.384710: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Trainable parameters: 1504065
2026-02-27 16:14:38.865024: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
2026-02-27 16:14:48.433310: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2026-02-27 16:14:48.492535: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8907
2134/2134 [==============================] - 134s 59ms/step - loss: 0.9392 - val_loss: 1.1546
Epoch 2/10
2134/2134 [==============================] - 127s 59ms/step - loss: 0.8729 - val_loss: 1.1817
Epoch 3/10
2134/2134 [==============================] - 128s 60ms/step - loss: 0.8821 - val_loss: 1.2682
Epoch 4/10
2134/2134 [==============================] - 127s 59ms/step - loss: 0.8927 - val_loss: 1.2405
Epoch 5/10
2134/2134 [==============================] - 126s 59ms/step - loss: 0.9015 - val_loss: 1.2895
Epoch 6/10
2134/2134 [==============================] - 128s 60ms/step - loss: 0.9113 - val_loss: 1.2121
Epoch 7/10
2134/2134 [==============================] - 127s 60ms/step - loss: 0.9194 - val_loss: 1.1449
Epoch 8/10
2134/2134 [==============================] - 126s 59ms/step - loss: 0.9282 - val_loss: 1.0968
Epoch 9/10
2134/2134 [==============================] - 127s 60ms/step - loss: 0.9341 - val_loss: 1.1098
Epoch 10/10
2134/2134 [==============================] - 126s 59ms/step - loss: 0.9417 - val_loss: 1.0730
2026-02-27 16:35:58.455143: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.
2026-02-27 16:35:58.823706: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.

Test metrics (original scale):
  Step-1 RMSE:  208.2149
  Step-1 MAPE:  11.5105%
  Horizon RMSE: 206.9925
  Horizon MAPE: 11.4357%

Saved predictions: outputs\predictions_unet_L168_H24.csv
Saved model:       outputs\model_unet_L168_H24
WARNING:absl:Found untraced functions such as token_embedding_layer_call_fn, token_embedding_layer_call_and_return_conditional_losses, positional_encoding_layer_call_fn, positional_encoding_layer_call_and_return_conditional_losses, convbr_layer_layer_call_fn while saving (showing 5 of 185). These functions will not be directly callable after loading.
Saved keras model: outputs\keras_savedmodel_unet_L168_H24
Saved plot:        outputs\actual_vs_pred_unet.png