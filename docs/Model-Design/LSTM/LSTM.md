## Overview
We will use a Long Short-Term Memory (LSTM) recurrent neural network (RNN) model to exploit the temporal statistics of our dataset. LSTMs have been shown to outperform time-blind classifiers such as convolutional neural networks (CNNs) in EEG classification tasks.<sup>[[1](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)]</sup> However, a CNN might still prove useful as a feature extraction step before input to the LSTM.

Unlike most EEG seizure prediction tasks, which usually suffer from a heavy class imbalance, our method can produce any desired balance of classes by adjusting the sizes of the of the pre and post-recovery windows. However, the reliability of the focus label degrades as the post-recovery window is extended. A meditator's focus can degrade surprisingly quickly. We will therefore need to test the effect of window size on performance, and we expect the optimal post-recovery window size to be in the 2-5 second range.

An LSTM model's input is a sequence of samples. While the optimal sequence size will need to be explored, we might have an upper bound imposed by practical considerations. With a 5-second window and 0.5-second samples, we can extract at most 10 sequential samples. The sequence size would be even smaller with larger sample sizes, which might be necessary for either the model or the featurization process to capture relevant temporal dynamics. One way to circumvent this problem while also possibly avoiding model overfitting would be to use the shuffling technique employed by Tsiouris et al., whereby a batch is constructed of non-consecutive samples. However, the model of Tsiouris et al. uses engineered features as its input, so it remains to be seen if this technique generalizes well to a model which takes the EEG signals themselves as input.

We will test three different variants of LSTM networks.

### LSTM
TODO

### CNN + LSTM
TODO

### LSTM with Engineered Features

Tsiouris et al. showed that model performance can improve dramatically when trained on featurized data over raw data alone. We will replicate their LSTM network and feature extraction. We will also this approach a step further by attempting to identify which specific features most improved model performance, an analysis which was missing from the work by Tsiouris et al..

For this model, we use a 1-second sample size. This larger size was chosen so that frequency domain features can capture. The below features are then extracted from each sample and used as the input to the LSTM. In other words, the LSTM does not receive any raw EEG data as direct input. We then shuffle the samples and construct 10-sample sequences as input to the LSTM. As in the original work, the LSTM has two layers of 128 units each. These are followed by a 30-unit fully-connected layer using ReLU activation, and finally a single-unit output layer using sigmoid activation. Also as in the original work, dropout and batch normalization are not used anywhere in the network.

| Feature Type     | Feature                                       | Count                                | Notes                                                       |
| ---------------- | --------------------------------------------- | ------------------------------------ | ------------------------------------------------------------|
| Time domain      | Mean                                          | 1 per channel                        |                                                             |
| Time domain      | Variance                                      | 1 per channel                        |                                                             |
| Time domain      | Standard Deviation                            | 1 per channel                        |                                                             |
| Time domain      | Skew                                          | 1 per channel                        |                                                             |
| Time domain      | Kurtosis                                      | 1 per channel                        |                                                             |
| Time domain      | Number of zero-crossings                      | 1 per channel                        |                                                             |
| Time domain      | Difference between maximum and minimum values | 1 per channel                        |                                                             |
| Time domain      | Absolute area under curve                     | 1 per channel                        |                                                             |
| Frequency domain | Total power                                   | 1 per channel                        |                                                             |
| Frequency domain | % power in Delta band                         | 1 per channel                        | 0Hz $\le f \lt$ 4Hz                                         |
| Frequency domain | % power in Theta band                         | 1 per channel                        | 4Hz $\le f \lt$ 8Hz                                         |
| Frequency domain | % power in Alpha band                         | 1 per channel                        | 8Hz $\le f \lt$ 14Hz                                        |
| Frequency domain | % power in Beta band                          | 1 per channel                        | 14Hz $\le f \lt$ 30Hz                                       |
| Frequency domain | % power in Gamma1 band                        | 1 per channel                        | 30Hz $\le f \lt$ 65Hz                                       |
| Frequency domain | % power in Gamma2 band                        | 1 per channel                        | 65Hz $\le f \lt$ 110Hz                                      |
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

TODO: Diagram

## Diagrams
[[resources/lstm_model.png]]

## References
1. [Tsiouris, Κ. Μ., Pezoulas, V. C., Zervakis, M., Konitsiotis, S., Koutsouris, D. D., & Fotiadis, D. I. (2018). A Long Short-Term Memory deep learning network for the prediction of epileptic seizures using EEG signals. *Computers in Biology and Medicine*, 99, 24–37. doi: 10.1016/j.compbiomed.2018.05.019](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)
