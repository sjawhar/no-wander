## Overview
We will use a Long Short-Term Memory (LSTM) recurrent neural network (RNN) model to exploit the temporal statistics of our dataset. LSTMs have been shown to outperform time-blind classifiers such as convolutional neural networks (CNNs) in EEG classification tasks.<sup>[[1](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)]</sup> However, a CNN might still prove useful as a feature extraction step before input to the LSTM.

Unlike most EEG seizure prediction tasks, which usually suffer from a heavy class imbalance, our method can produce any desired balance of classes by adjusting the sizes of the of the pre and post-recovery windows. However, the reliability of the focus label degrades as the post-recovery window is extended. A meditator's focus can degrade surprisingly quickly. We will therefore need to test the effect of window size on performance, and we expect the optimal post-recovery window size to be in the 2-5 second range.

An LSTM model's input is a sequence of segments. While the optimal sequence size will need to be explored, we might have an upper bound imposed by practical considerations. With a 5-second window and 0.5-second segments, we can extract at most 10 sequential segments. The sequence size would be even smaller with larger segment sizes, which might be necessary for either the model or the featurization process to capture relevant temporal dynamics. One way to circumvent this problem while also possibly avoiding model overfitting would be to use the shuffling technique employed by Tsiouris et al., whereby a batch is constructed of non-consecutive segments. However, the model of Tsiouris et al. uses engineered features as its input, so it remains to be seen if this technique generalizes well to a model which takes the EEG signals themselves as input.

We will test three different variants of LSTM networks.

### LSTM
The first model variant is a pure LSTM. We will experiment with the number of units (32, 64, 128) and layers (one or two) to ensure the model has enough capacity to fully represent the task being learned. The final (or only) LSTM layer feeds to a 32-unit fully-connected layer using ReLU activation followed by a single-unit output sigmoid layer. Dropout is used on the input to the LSTM, and each LSTM and fully-connected layer is followed by an "independent-component (IC) layer,"<sup>[[2](https://arxiv.org/abs/1905.05928)]</sup> actually a combination of batch normalization and dropout layers. A dropout rate of 0.2 is used for all dropout layers.

The input to this network is the raw, unprocessed data recorded from the Muse 2. This includes EEG, PPG, accelerometer, and gyroscope readings. Epochs are split into 0.5 second epochs with pre, post, and uncertainty window sizes of 6, 3, and 1, respectively. Grouping segments into 6-segment sequences yields two distraction sequences and one focus sequence per epoch.

[[resources/model_lstm.png]]

### LSTM with CNN
The second model variant is very similar to the pure LSTM variant, except that it includes a 1-D convolutional layer with 128 filters (the CNN) between the raw input and the LSTM input. The input to this network is the same as the pure LSTM, except that the full 3-second sequence is fed as input to the CNN instead of chunking into sequences for the LSTM. This produces an input sequence to the LSTM whose size depends on the parameters of the CNN (number of filters, stride sizes).

The benefits of the CNN are to reduce the dimensionality of the input to the LSTM, and therefore reduces the number of trainable parameters, while also extracting time-invariant features of the input. The drawback is that it increases the LSTM sequence length, which can reduce model performance.

[[resources/model_lstm_cnn.png]]

### LSTM with Engineered Features

Tsiouris et al. showed that model performance can improve dramatically when trained on featurized data over raw data alone. We have replicated their LSTM network and feature extraction pipeline (see [Featurization](Featurization) for more details). We extend this approach by attempting to identify which specific features most improved model performance, an analysis which was missing from the work by Tsiouris et al..

For this model, we use a 1-second segment size. This larger size was chosen so that frequency domain features can be captured. The features are then extracted from each segment and used as the input to the LSTM. In other words, the LSTM does not receive any raw EEG data as direct input. We then shuffle the segments and construct 10-segment sequences as input to the LSTM. As in the original work, the LSTM has two layers of 128 units each. These are followed by a 30-unit fully-connected layer using ReLU activation, and finally a single-unit output layer using sigmoid activation. Also as in the original work, dropout and batch normalization are not used anywhere in the network.

[[resources/model_lstm_feature.png]]

## Hyperparameter Search
The performance of LSTM networks is notoriously sensitive to the choice of hyperparameters (e.g. number of layers, number of units, even batch size). Zhang et al. (2017)<sup>[[3](https://link.springer.com/chapter/10.1007/978-3-319-70096-0_76)]</sup> leveraged a software testing technique called orthogonal array (OA) testing to design a 7-layer RNN classifier for EEG input, with promising results. We will also explore this technique in testing our own models.

## References
1. [Tsiouris, Κ. Μ., Pezoulas, V. C., Zervakis, M., Konitsiotis, S., Koutsouris, D. D., & Fotiadis, D. I. (2018). A Long Short-Term Memory deep learning network for the prediction of epileptic seizures using EEG signals. *Computers in Biology and Medicine*, 99, 24–37. doi: 10.1016/j.compbiomed.2018.05.019](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)
2. [Chen, G., Chen, P., Shi, Y., Hsieh, C. Y., Liao, B., & Zhang, S. (2019). Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks. *arXiv preprint arXiv:1905.05928*.](https://arxiv.org/abs/1905.05928)
3. [Zhang, X., Yao, L., Huang, C., Sheng, Q. Z., & Wang, X. (2017, November). Intent recognition in smart living through deep recurrent neural networks. In *International Conference on Neural Information Processing* (pp. 748-758). Springer, Cham.](https://link.springer.com/chapter/10.1007/978-3-319-70096-0_76)