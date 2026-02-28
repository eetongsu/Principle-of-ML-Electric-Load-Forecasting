Model: transformer
L=168, H=24, stride=1
Rows: train=136719, val=15191, test=37978
Windows: train=136528, val=15000, test=37980
Predict batch size: 256
=============================
2026-02-27 17:59:51.763197: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Trainable parameters: 2782465
2026-02-27 18:00:09.165530: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 3211138560 exceeds 10% of free system memory.
Epoch 1/10
2026-02-27 18:02:40.391621: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
2134/2134 [==============================] - 2261s 990ms/step - loss: 1.1426 - val_loss: 0.9810
Epoch 2/10
2134/2134 [==============================] - 2093s 981ms/step - loss: 1.0025 - val_loss: 0.9812
Epoch 3/10
2134/2134 [==============================] - 2041s 956ms/step - loss: 1.0012 - val_loss: 0.9812
Epoch 4/10
2134/2134 [==============================] - 1973s 925ms/step - loss: 1.0028 - val_loss: 0.9852
Epoch 5/10
2134/2134 [==============================] - 2030s 951ms/step - loss: 1.0007 - val_loss: 0.9812
Epoch 6/10
2134/2134 [==============================] - 1972s 924ms/step - loss: 1.0002 - val_loss: 0.9810
Epoch 7/10
2134/2134 [==============================] - 2061s 966ms/step - loss: 1.0002 - val_loss: 0.9810
Epoch 8/10
2134/2134 [==============================] - 2058s 964ms/step - loss: 1.0002 - val_loss: 0.9811
Epoch 9/10
2134/2134 [==============================] - 2021s 947ms/step - loss: 1.0002 - val_loss: 0.9811
Epoch 10/10
2134/2134 [==============================] - 2096s 982ms/step - loss: 1.0002 - val_loss: 0.9811
2026-02-27 23:43:38.561067: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.
2026-02-27 23:43:38.948201: W tensorflow/core/framework/cpu_allocator_impl.cc:82] Allocation of 893289600 exceeds 10% of free system memory.

Test metrics (original scale):
  Step-1 RMSE:  199.4985
  Step-1 MAPE:  11.0359%
  Horizon RMSE: 199.4968
  Horizon MAPE: 11.0361%

Saved predictions: outputs\predictions_transformer_L168_H24.csv
Saved model:       outputs\model_transformer_L168_H24
WARNING:absl:Found untraced functions such as token_embedding_layer_call_fn, token_embedding_layer_call_and_return_conditional_losses, positional_encoding_layer_call_fn, positional_encoding_layer_call_and_return_conditional_losses, data_embedding_layer_call_fn while saving (showing 5 of 194). These functions will not be directly callable after loading.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72BF5120> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72BDEA70> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72CC6320> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72C42380> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72C437F0> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72C403D0> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C715A37C0> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72BB2EC0> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C72BDF910> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
WARNING:absl:<tfts.layers.attention_layer.Attention object at 0x0000020C7165DC90> has the same name 'Attention' as a built-in Keras object. Consider renaming <class 'tfts.layers.attention_layer.Attention'> to avoid naming conflicts when loading with `tf.keras.models.load_model`. If renaming is not possible, pass the object in the `custom_objects` parameter of the load function.
Saved keras model: outputs\keras_savedmodel_transformer_L168_H24
Saved plot:        outputs\actual_vs_pred_transformer.png