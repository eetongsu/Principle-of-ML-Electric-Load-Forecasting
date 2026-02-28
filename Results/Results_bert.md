## Model: bert

L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980

Predict batch size: 256

2026-02-28 01:37:43.904666: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Trainable parameters: 676760
2026-02-28 01:37:47.845074: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
2026-02-28 01:37:54.778693: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2134/2134 [==============================] - 53s 23ms/step - loss: 1.0011 - val_loss: 0.9812
Epoch 2/10
2134/2134 [==============================] - 49s 23ms/step - loss: 1.0003 - val_loss: 0.9811
Epoch 3/10
2134/2134 [==============================] - 48s 23ms/step - loss: 1.0003 - val_loss: 0.9812
Epoch 4/10
2134/2134 [==============================] - 49s 23ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 5/10
2134/2134 [==============================] - 47s 22ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 6/10
2134/2134 [==============================] - 48s 23ms/step - loss: 1.0004 - val_loss: 0.9812
Epoch 7/10
2134/2134 [==============================] - 45s 21ms/step - loss: 1.0003 - val_loss: 0.9812
Epoch 8/10
2134/2134 [==============================] - 55s 26ms/step - loss: 1.0003 - val_loss: 0.9811
Epoch 9/10
2134/2134 [==============================] - 57s 26ms/step - loss: 1.0003 - val_loss: 0.9812
Epoch 10/10
2134/2134 [==============================] - 57s 27ms/step - loss: 1.0003 - val_loss: 0.9811
2026-02-28 01:46:18.778913: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.
2026-02-28 01:46:19.181138: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.

Test metrics (original scale):
  Step-1 RMSE:  199.4808
  Step-1 MAPE:  11.0201%
  Horizon RMSE: 199.4927
  Horizon MAPE: 11.0263%

Saved predictions: outputs\predictions_bert_L168_H24.csv
Saved model:       outputs\model_bert_L168_H24
WARNING:absl:Found untraced functions such as token_embedding_layer_call_fn, token_embedding_layer_call_and_return_conditional_losses, positional_encoding_layer_call_fn, positional_encoding_layer_call_and_return_conditional_losses, self_attention_layer_call_fn while saving (showing 5 of 48). These functions will not be directly callable after loading.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x00000244BF9BB370> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x00000244C106D810> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
Saved keras model: outputs\keras_savedmodel_bert_L168_H24
Saved plot:        outputs\actual_vs_pred_bert.png