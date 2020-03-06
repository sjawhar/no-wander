## Overview

Though an ideal machine learning (ML) model might be able to learn entirely from raw, labeled input, this is rarely the case in practice. More usually, a process of "featurizing" is required, whereby the data is transformed and manipulated such that relevant statistics can be more easily captured by the model. Such preprocessing techniques can vary from nearly ubiquitous (e.g. scaling channel-wise to zero mean and unit variance) to highly specialized. The latter type, though potentially very effective, often require significant domain knowledge and yet, as they are specifically engineered to a particular task, can be less robust. Here we discuss the methods employed by this project.

### Centering and scaling
As mentioned above, perhaps the most uniquitous preprocessing step in ML is to center and scale each channel to zero mean and unit variance. However, as EEG data is notoriously noisy and artifact-prone, we use a version of this technique that is more robust to outliers: the median is used instead of the mean for centering, and the variance is calculated using only the data between the first and third quartiles.

### EEG-specific features
Tsiouris et al. (2018)<sup>[[1](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)]</sup> explored the effectiveness of various features used across the EEG classification literature and showed them to improve the performance of an LSTM model over learning directly on raw input.

Note that these features can be reduced to reduce the dimensionality of the data, if they are used _in place of_ rather than in addition to the raw EEG recording. For example, the number of dimensions in a 1 second sample from a 32-channel, 256 Hz recording is reduced from 8,192 dimensions to 1,652.

#### Time domain
| Feature                                       | Count         |
| --------------------------------------------- | ------------- |
| Mean                                          | 1 per channel |
| Variance                                      | 1 per channel |
| Standard Deviation                            | 1 per channel |
| Skew                                          | 1 per channel |
| Kurtosis                                      | 1 per channel |
| Number of zero-crossings                      | 1 per channel |
| Difference between maximum and minimum values | 1 per channel |
| Absolute area under curve                     | 1 per channel |

#### Frequency domain
| Feature                                 | Count                                | Notes                                              |
| --------------------------------------- | ------------------------------------ | -------------------------------------------------- |
| Total power                             | 1 per channel                        |                                                    |
| % power in Delta band                   | 1 per channel                        | 0Hz ≤ f < 4Hz                                      |
| % power in Theta band                   | 1 per channel                        | 4Hz ≤ f < 8Hz                                      |
| % power in Alpha band                   | 1 per channel                        | 8Hz ≤ f < 14Hz                                     |
| % power in Beta band                    | 1 per channel                        | 14Hz ≤ f < 30Hz                                    |
| % power in Gamma1 band                  | 1 per channel                        | 30Hz ≤ f < 65Hz                                    |
| % power in Gamma2 band                  | 1 per channel                        | 65Hz ≤ f < 110Hz                                   |
| Discrete wavelet transform coefficients | 16 per channel (8 approx + 8 detail) | 7-level decomposition, Daubechies 4 mother wavelet |

#### Correlation
| Feature                                  | Count                   |
| ---------------------------------------- | ----------------------- |
| Maximum time-lagged pairwise correlation | Number of channel pairs |
| Decorrelation time                       | 1 per channel           |

The decorrelation time is the time to the first zero-crossing in a channel's autocorrelation function.

#### Graph theory
Each channel is a node in the sample's constructed graph, and the connection between nodes are weighted by the maximum time-lagged correlation between the corresponding channels in that sample.

| Feature                     | Count         |
| --------------------------- | ------------- |
| Node clustering coefficient | 1 per channel |
| Node efficiency             | 1 per channel |
| Node betweenness centrality | 1 per channel |
| Node eccentricity           | 1 per channel |
| Graph lambda                | 1 total       |
| Graph efficiency            | 1 total       |
| Graph radius                | 1 total       |
| Graph diameter              | 1 total       |

## References
1. [Tsiouris, Κ. Μ., Pezoulas, V. C., Zervakis, M., Konitsiotis, S., Koutsouris, D. D., & Fotiadis, D. I. (2018). A Long Short-Term Memory deep learning network for the prediction of epileptic seizures using EEG signals. *Computers in Biology and Medicine*, 99, 24–37. doi: 10.1016/j.compbiomed.2018.05.019](https://www.sciencedirect.com/science/article/pii/S001048251830132X#cebib0010)