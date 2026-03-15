# Electric-Load-Forecasting

## 1 Current Algorithm

### RNN

![actual_vs_pred_bert](outputs/actual_vs_pred_bert.png)

**RMSE**: 200.3891
**MAPE**: 11.0782%

### TCN

![actual_vs_pred_bert](outputs/actual_vs_pred_tcn.png)

**RMSE**: 199.5089
**MAPE**: 11.0325%

### WaveNet

![actual_vs_pred_bert](outputs/actual_vs_pred_wavenet.png)

**RMSE**: 88.1781
**MAPE**: 4.8761%

### UNet

![actual_vs_pred_bert](outputs/actual_vs_pred_unet.png)

**RMSE**: 216.2911
**MAPE**: 11.8973%

### Transformer

![actual_vs_pred_bert](outputs/actual_vs_pred_transformer.png)

**RMSE**: 199.4722
**MAPE**: 11.0150%

### Bert

![actual_vs_pred_bert](outputs/actual_vs_pred_bert.png)

**RMSE**: 199.5089
**MAPE**: 11.0325%

### TFT

![actual_vs_pred_bert](outputs/actual_vs_pred_tft.png)

**RMSE**: 195.3225
**MAPE**: 10.8010%

### N-BEATS

![actual_vs_pred_bert](outputs/actual_vs_pred_nbeats.png)

**RMSE**: 199.4748
**MAPE**: 11.0240%

### Informer

![actual_vs_pred_informer](outputs/actual_vs_pred_informer.png)

**RMSE**: 79.1236
**MAPE**: 4.3754%

## 2 Informer-Based Typical Days

![typical_day_20240614_0000](outputs_typical_days/typical_day_20240614_0000.png)

![typical_day_20250318_0000](outputs_typical_days/typical_day_20250102_0000.png)

![typical_day_20250529_0000](outputs_typical_days/typical_day_20250528_0000.png)

## 3 Conclusion

This study investigated the effectiveness of several deep learning architectures for short-term load forecasting, including RNN, TCN, WaveNet, Transformer-based models, BERT, TFT, N-BEATS, and Informer. These models represent different paradigms for temporal sequence modeling, ranging from recurrent structures and convolutional temporal networks to attention-based architectures. Through systematic experimentation, their forecasting capability on complex load sequences was evaluated.

Traditional sequence models such as **RNN** rely on recurrent connections to capture temporal dependencies. While they can model sequential patterns, their ability to learn long-range dependencies is limited due to gradient vanishing and the inherently sequential nature of computation. As a result, their predictions tend to converge toward averaged patterns of the load sequence rather than capturing the full variability of the data.

**Temporal Convolutional Networks (TCN)** attempt to address this limitation using dilated causal convolutions, which allow the model to capture longer receptive fields compared with standard convolutional approaches. However, despite their improved parallelization and temporal receptive range, their capability to represent highly dynamic load fluctuations remains constrained when the underlying patterns are complex and non-stationary.

**WaveNet**, which is also based on dilated convolutional structures, performs better in modeling temporal dependencies because the stacked dilation layers expand the receptive field exponentially. This enables the model to capture both short-term and medium-term dependencies in the load sequence. Consequently, WaveNet demonstrates a significant improvement over standard recurrent or convolutional architectures.

Attention-based architectures, such as the **Transformer**, introduce self-attention mechanisms that allow the model to directly learn dependencies between any two time steps in the sequence. Although this design theoretically enables strong representation of long-term dependencies, the quadratic computational complexity of the self-attention mechanism limits its efficiency when dealing with long time-series inputs. In practice, this may lead to suboptimal performance when the sequence length increases or when the data contains large amounts of noise.

Models adapted from natural language processing, such as **BERT**, rely on bidirectional attention structures. While this architecture is powerful in representation learning, it is not specifically designed for forecasting tasks, particularly those requiring autoregressive prediction of future values. Consequently, its ability to generalize to time-series forecasting scenarios is somewhat restricted.

The **Temporal Fusion Transformer (TFT)** integrates attention mechanisms with gating and variable selection networks, enabling the model to incorporate both static and dynamic features. Although this architecture enhances interpretability and improves feature utilization, its structural complexity increases the training difficulty and may limit its ability to fully exploit long sequence dependencies under certain conditions.

**N-BEATS** adopts a deep fully connected architecture with backward and forward residual connections to decompose time-series signals into interpretable components. While it has demonstrated strong performance in many forecasting tasks, its reliance on stacked fully connected blocks may reduce its ability to capture intricate temporal interactions when the data exhibits highly irregular patterns.

Among all evaluated approaches, **Informer** demonstrates the most effective performance. Informer is specifically designed for long sequence time-series forecasting and introduces several architectural innovations. First, it employs the **ProbSparse self-attention mechanism**, which significantly reduces computational complexity by focusing on the most informative query-key pairs. This allows the model to maintain the advantages of attention-based learning while improving efficiency for long sequences. Second, Informer incorporates a **self-attention distilling strategy**, which progressively reduces the sequence length through convolution and pooling operations. This mechanism compresses redundant temporal information and enhances the representation of essential features. Finally, Informer adopts a **generative inference strategy**, enabling the model to predict the entire future sequence in a single forward pass rather than generating predictions step by step.

These design choices allow Informer to effectively capture both global dependencies and local temporal structures within the load data. By selectively focusing on important temporal interactions and reducing redundant information, the model achieves a more accurate representation of the underlying load dynamics. Consequently, Informer produces predictions that more closely follow the actual fluctuations of the load sequence compared with other architectures.

Overall, the comparative analysis demonstrates that architectures specifically designed for long sequence forecasting significantly outperform traditional recurrent, convolutional, and general-purpose attention-based models. The results highlight the importance of efficient attention mechanisms and sequence distillation strategies for modeling complex temporal patterns in energy load forecasting. Future research may further improve forecasting performance by integrating domain-specific features, hybrid architectures, or adaptive attention mechanisms tailored for energy systems.

