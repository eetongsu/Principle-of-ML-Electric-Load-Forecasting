## Model: WaveNet

L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980

Predict batch size: 256

2026-02-28 14:15:07.604483: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Trainable parameters: 470274
2026-02-28 14:15:13.120103: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
WARNING:tensorflow:Gradients do not exist for variables ['encoder/conv_temp_3/kernel:0', 'encoder/conv_temp_3/bias:0', 'encoder/encoder_dense_time3/kernel:0', 'encoder/encoder_dense_time3/bias:0', 'encoder/encoder_dense_time_4/kernel:0', 'encoder/encoder_dense_time_4/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
WARNING:tensorflow:Gradients do not exist for variables ['encoder/conv_temp_3/kernel:0', 'encoder/conv_temp_3/bias:0', 'encoder/encoder_dense_time3/kernel:0', 'encoder/encoder_dense_time3/bias:0', 'encoder/encoder_dense_time_4/kernel:0', 'encoder/encoder_dense_time_4/bias:0'] when minimizing the loss. If you're using `model.compile()`, did you forget to provide a `loss` argument?
2026-02-28 14:15:32.765343: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2026-02-28 14:15:32.826823: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8907
2134/2134 [==============================] - 296s 130ms/step - loss: 0.0043 - val_loss: 0.0017
Epoch 2/10
2134/2134 [==============================] - 274s 129ms/step - loss: 2.7588e-04 - val_loss: 2.2606e-04
Epoch 3/10
2134/2134 [==============================] - 274s 128ms/step - loss: 3.8344e-04 - val_loss: 2.3801e-04
Epoch 4/10
2134/2134 [==============================] - 272s 127ms/step - loss: 2.5931e-04 - val_loss: 0.0014
Epoch 5/10
2134/2134 [==============================] - 283s 133ms/step - loss: 2.1679e-04 - val_loss: 2.7867e-04
Epoch 6/10
2134/2134 [==============================] - 274s 128ms/step - loss: 1.8875e-04 - val_loss: 1.4358e-04
Epoch 7/10
2134/2134 [==============================] - 272s 128ms/step - loss: 1.1567e-04 - val_loss: 9.1377e-05
Epoch 8/10
2134/2134 [==============================] - 270s 126ms/step - loss: 9.9712e-05 - val_loss: 4.4533e-05
Epoch 9/10
2134/2134 [==============================] - 272s 127ms/step - loss: 9.0236e-05 - val_loss: 7.3893e-05
Epoch 10/10
2134/2134 [==============================] - 268s 126ms/step - loss: 5.6323e-05 - val_loss: 1.2065e-04
2026-02-28 15:01:12.148065: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.
2026-02-28 15:01:12.506677: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.

Test metrics (original scale):
  Step-1 RMSE:  2.8237
  Step-1 MAPE:  0.1335%
  Horizon RMSE: 2.7076
  Horizon MAPE: 0.1296%

Saved predictions: outputs\predictions_wavenet_L168_H24.csv
ERROR:root:Failed to save model weights to outputs\model_wavenet_L168_H24\tf_model.weights.h5: Unable to synchronously create dataset (name already exists)
Saved model:       outputs\model_wavenet_L168_H24
Saved keras model: outputs\keras_savedmodel_wavenet_L168_H24
Saved plot:        outputs\actual_vs_pred_wavenet.png