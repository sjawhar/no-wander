## Overview
We will use a Long Short-Term Memory (LSTM) recurrent neural network (RNN) model to exploit the temporal statistics of our dataset. LSTMs have been shown to outperform time-blind classifiers such as convolutional neural networks (CNNs) in EEG classification tasks.<sup>[[1](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)]</sup> However, a CNN might still prove useful as a feature extraction step before input to the LSTM.

Unlike most EEG seizure prediction tasks, which usually suffer from a heavy class imbalance, our method can produce any desired balance of classes by adjusting the sizes of the of the pre and post-recovery windows. However, the reliability of the focus label degrades as the post-recovery window is extended. A meditator's focus can degrade surprisingly quickly. We will therefore need to test the effect of window size on performance, and we expect the optimal post-recovery window size to be in the 2-5 second range.

An LSTM model's input is a sequence of samples. While the optimal sequence size will need to be explored, we might have an upper bound imposed by practical considerations. With a 5-second window and 0.5-second samples, we can extract at most 10 sequential samples. The sequence size would be even smaller with larger sample sizes, which might be necessary for either the model or the featurization process to capture relevant temporal dynamics. One way to circumvent this problem while also possibly avoiding model overfitting would be to use the shuffling technique employed by Tsiouris et al., whereby a batch is constructed of non-consecutive samples. However, the model of Tsiouris et al. uses engineered features as its input, so it remains to be seen if this technique generalizes well to a model which takes the EEG signals themselves as input.

We will test three different variants of LSTM networks.

### LSTM
The first model variant is a pure LSTM. We will experiment with the number of units (32, 64, 128) and layers (one or two) to ensure the model has enough capacity to fully represent the task being learned. The final (or only) LSTM layer feeds to a 32-unit fully-connected layer using ReLU activation followed by a single-unit output sigmoid layer. Dropout is used on the input to the LSTM, and each LSTM and fully-connected layer is followed by an "independent-component (IC) layer,"<sup>[[2](https://arxiv.org/abs/1905.05928)]</sup> actually a combination of batch normalization and dropout layers. A dropout rate of 0.2 is used for all dropout layers.

The input to this network is the raw, unprocessed data recorded from the Muse 2. This includes EEG, PPG, accelerometer, and gyroscope readings. Epochs are split into 0.5 second epochs with pre, post, and uncertainty window sizes of 6, 3, and 1, respectively. Grouping samples into 6-sample sequences yields two distraction sequences and one focus sequence per epoch.

[[resources/model_lstm.png]]

### LSTM with CNN
The second model variant is very similar to the pure LSTM variant, except that it includes a 1-D convolutional layer with 128 filters (the CNN) between the raw input and the LSTM input. The input to this network is the same as the pure LSTM, except that the full 3-second sequence is fed as input to the CNN instead of chunking into sequences for the LSTM. This produces an input sequence to the LSTM whose size depends on the parameters of the CNN (number of filters, stride sizes).

The benefits of the CNN are to reduce the dimensionality of the input to the LSTM, and therefore reduces the number of trainable parameters, while also extracting time-invariant features of the input. The drawback is that it increases the LSTM sequence length, which can reduce model performance.

[[resources/model_lstm_cnn.png]]

### LSTM with Engineered Features

Tsiouris et al. showed that model performance can improve dramatically when trained on featurized data over raw data alone. We ahve replicated their LSTM network and feature extraction. We extend this approach by attempting to identify which specific features most improved model performance, an analysis which was missing from the work by Tsiouris et al..

For this model, we use a 1-second sample size. This larger size was chosen so that frequency domain features can capture. The below features are then extracted from each sample and used as the input to the LSTM. In other words, the LSTM does not receive any raw EEG data as direct input. We then shuffle the samples and construct 10-sample sequences as input to the LSTM. As in the original work, the LSTM has two layers of 128 units each. These are followed by a 30-unit fully-connected layer using ReLU activation, and finally a single-unit output layer using sigmoid activation. Also as in the original work, dropout and batch normalization are not used anywhere in the network.

| Feature Type     | Feature                                       | Count                                | Notes                                                       |
| ---------------- | --------------------------------------------- | ------------------------------------ | ----------------------------------------------------------- |
| Time domain      | Mean                                          | 1 per channel                        |                                                             |
| Time domain      | Variance                                      | 1 per channel                        |                                                             |
| Time domain      | Standard Deviation                            | 1 per channel                        |                                                             |
| Time domain      | Skew                                          | 1 per channel                        |                                                             |
| Time domain      | Kurtosis                                      | 1 per channel                        |                                                             |
| Time domain      | Number of zero-crossings                      | 1 per channel                        |                                                             |
| Time domain      | Difference between maximum and minimum values | 1 per channel                        |                                                             |
| Time domain      | Absolute area under curve                     | 1 per channel                        |                                                             |
| Frequency domain | Total power                                   | 1 per channel                        |                                                             |
| Frequency domain | % power in Delta band                         | 1 per channel                        | 0Hz ≤ f < 4Hz                                               |
| Frequency domain | % power in Theta band                         | 1 per channel                        | 4Hz ≤ f < 8Hz                                               |
| Frequency domain | % power in Alpha band                         | 1 per channel                        | 8Hz ≤ f < 14Hz                                              |
| Frequency domain | % power in Beta band                          | 1 per channel                        | 14Hz ≤ f < 30Hz                                             |
| Frequency domain | % power in Gamma1 band                        | 1 per channel                        | 30Hz ≤ f < 65Hz                                             |
| Frequency domain | % power in Gamma2 band                        | 1 per channel                        | 65Hz ≤ f < 110Hz                                            |
| Frequency domain | Discrete wavelet transform coefficients       | 16 per channel (8 approx + 8 detail) | 7-level decomposition, Daubechies 4 mother wavelet          |
| Correlation      | Maximum time-lagged pairwise correlation      | Number of channel pairs              |                                                             |
| Correlation      | Decorrelation time                            | 1 per channel                        | Time to first zero-crossing in autocorrelation function     |
| Graph theory     | Node clustering coefficient                   | 1 per channel                        | Nodes are channels, weights are max time-lagged correlation |
| Graph theory     | Node efficiency                               | 1 per channel                        | See above                                                   |
| Graph theory     | Node betweenness centrality                   | 1 per channel                        | See above                                                   |
| Graph theory     | Node eccentricity                             | 1 per channel                        | See above                                                   |
| Graph theory     | Graph lambda                                  | 1                                    | See above                                                   |
| Graph theory     | Graph efficiency                              | 1                                    | See above                                                   |
| Graph theory     | Graph radius                                  | 1                                    | See above                                                   |
| Graph theory     | Graph diameter                                | 1                                    | See above                                                   |

[[resources/model_lstm_feature.png]]

## References
1. [Tsiouris, Κ. Μ., Pezoulas, V. C., Zervakis, M., Konitsiotis, S., Koutsouris, D. D., & Fotiadis, D. I. (2018). A Long Short-Term Memory deep learning network for the prediction of epileptic seizures using EEG signals. *Computers in Biology and Medicine*, 99, 24–37. doi: 10.1016/j.compbiomed.2018.05.019](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)
2. [Chen, G., Chen, P., Shi, Y., Hsieh, C. Y., Liao, B., & Zhang, S. (2019). Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks. *arXiv preprint arXiv:1905.05928*.](https://arxiv.org/abs/1905.05928)