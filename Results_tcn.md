## Results

Model: tcn
L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980

Predict batch size: 256
=============================
2026-02-27 13:58:23.722048: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Trainable parameters: 917017
2026-02-27 13:58:27.025766: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
WARNING:tensorflow:Gradients do not exist for variables ['encoder/kernel:0', 'encoder/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['encoder/kernel:0', 'encoder/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
2026-02-27 13:58:33.033786: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2026-02-27 13:58:33.108020: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8907
2134/2134 [==============================] - 36s 15ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 2/10
2134/2134 [==============================] - 30s 14ms/step - loss: 1.0004 - val_loss: 0.9811
Epoch 3/10
2134/2134 [==============================] - 31s 15ms/step - loss: 1.0003 - val_loss: 0.9812
Epoch 4/10
2134/2134 [==============================] - 30s 14ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 5/10
2134/2134 [==============================] - 31s 14ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 6/10
2134/2134 [==============================] - 31s 14ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 7/10
2134/2134 [==============================] - 31s 14ms/step - loss: 1.0003 - val_loss: 0.9812
Epoch 8/10
2134/2134 [==============================] - 31s 15ms/step - loss: 1.0003 - val_loss: 0.9811
Epoch 9/10
2134/2134 [==============================] - 32s 15ms/step - loss: 1.0003 - val_loss: 0.9812
Epoch 10/10
2134/2134 [==============================] - 30s 14ms/step - loss: 1.0003 - val_loss: 0.9811

Test metrics (original scale):
  Step-1 RMSE:  199.4808
  Step-1 MAPE:  11.0201%
  Horizon RMSE: 199.4927
  Horizon MAPE: 11.0263%

Saved predictions: outputs\predictions_tcn_L168_H24.csv
ERROR:root:Failed to save model weights to outputs\model_tcn_L168_H24\tf_model.weights.h5: Unable to synchronously create dataset (name already exists)
Saved model:       outputs\model_tcn_L168_H24
WARNING:absl:Found untraced functions such as encoder_dense_time1_layer_call_fn, encoder_dense_time1_layer_call_and_return_conditional_losses, encoder_dense_time2_layer_call_fn, encoder_dense_time2_layer_call_and_return_conditional_losses, encoder_dense_time3_layer_call_fn while saving (showing 5 of 28). These functions will not be directly callable after loading.
Saved keras model: outputs\keras_savedmodel_tcn_L168_H24
Saved plot:        outputs\actual_vs_pred_tcn.png