## Overview
We will use a Long Short-Term Memory (LSTM) recurrent neural network (RNN) model to exploit the temporal statistics of our dataset. LSTMs have been shown to outperform time-blind classifiers such as convolutional neural networks (CNNs) in EEG classification tasks.<sup>[[1](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)]</sup> However, a CNN might still prove useful as a feature extraction step before input to the LSTM. In fact, [1] showed that model performance can improve dramatically when trained on featurized data over raw data alone. This project will take this a step further by attempting to identify which specific features were most predictive of distraction.

Unlike [1], which suffered from a heavy class imbalance, our method can produce any desired balance of classes by adjusting the sizes of the of the pre and post-recovery windows. However, the reliability of the focus label degrades as the post-recovery window is extended. A meditator's focus can degrade surprisingly quickly. We will therefore need to test the effect of window size on performance, and we expect the optimal post-recovery window size to be in the 2-5 second range.

The model's input is a "batch" of samples. While the optimal batch size will need to be explored, we might be limited in the maximum batch size. With a 5-second window and 0.5-second samples, we can have at most 10 sequential samples in a batch. The batch size would be even smaller with larger sample sizes, which might be necessary for either the model or the featurization process to capture relevant temporal dynamics. One way to circumvent this problem while also possibly avoiding model overfitting would be to use the shuffling technique employed by [1], whereby a batch is constructed of non-consecutive samples.

## Diagrams
[[resources/lstm_model.png]]

## References
1. [Tsiouris, Κ. Μ., Pezoulas, V. C., Zervakis, M., Konitsiotis, S., Koutsouris, D. D., & Fotiadis, D. I. (2018). A Long Short-Term Memory deep learning network for the prediction of epileptic seizures using EEG signals. *Computers in Biology and Medicine*, 99, 24–37. doi: 10.1016/j.compbiomed.2018.05.019](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)
