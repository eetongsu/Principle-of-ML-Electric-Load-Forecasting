## Results

Model: tft
L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980

Predict batch size: 256

2026-02-26 22:19:55.886721: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Trainable parameters: 1726209
2026-02-26 22:19:59.919203: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
2026-02-26 22:20:08.094727: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2026-02-26 22:20:08.216677: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8907
2134/2134 [==============================] - 113s 50ms/step - loss: 0.5056 - val_loss: 3.4272e-04
Epoch 2/10
2134/2134 [==============================] - 105s 49ms/step - loss: 5.9907e-04 - val_loss: 5.6947e-04
Epoch 3/10
2134/2134 [==============================] - 105s 49ms/step - loss: 0.0656 - val_loss: 0.7561
Epoch 4/10
2134/2134 [==============================] - 108s 50ms/step - loss: 0.0346 - val_loss: 5.9677e-04
Epoch 5/10
2134/2134 [==============================] - 106s 50ms/step - loss: 0.0385 - val_loss: 0.8096
Epoch 6/10
2134/2134 [==============================] - 110s 52ms/step - loss: 0.1040 - val_loss: 0.0085
Epoch 7/10
2134/2134 [==============================] - 110s 51ms/step - loss: 0.1209 - val_loss: 0.8626
Epoch 8/10
2134/2134 [==============================] - 109s 51ms/step - loss: 0.2551 - val_loss: 0.0897
Epoch 9/10
2134/2134 [==============================] - 112s 52ms/step - loss: 0.1331 - val_loss: 0.3268
Epoch 10/10
2134/2134 [==============================] - 103s 48ms/step - loss: 0.1159 - val_loss: 0.0621
2026-02-26 22:38:03.634242: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.
2026-02-26 22:38:03.972844: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.

Test metrics (original scale):
  Step-1 RMSE:  49.3863
  Step-1 MAPE:  0.9998%
  Horizon RMSE: 45.8023
  Horizon MAPE: 1.0788%

Saved predictions: outputs\predictions_tft_L168_H24.csv
Saved model:       outputs\model_tft_L168_H24
WARNING:absl:Found untraced functions such as token_embedding_layer_call_fn, token_embedding_layer_call_and_return_conditional_losses, positional_encoding_layer_call_fn, positional_encoding_layer_call_and_return_conditional_losses, token_embedding_layer_call_fn while saving (showing 5 of 28). These functions will not be directly callable after loading.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000021E0D024AF0> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
Saved keras model: outputs\keras_savedmodel_tft_L168_H24
Saved plot:        outputs\actual_vs_pred_tft.png