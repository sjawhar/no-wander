## Overview
In sequence modeling tasks, attention is a technique for making more direct use of temporally-distant information by considering the entire input sequence at each timestep. In practice, this helps to capture long-range temporal correlations as well as resolve ambiguities (especially in language translation tasks).

## Multi-Head Attention
As indicated by the title of the paper that introduced it (Vaswani et al., 2017)<sup>[[1](http://papers.nips.cc/paper/7181-attention-is-all-you-need)]</sup>, the Transformer model aims to replace all convolutional or recurrence operations in sequence modeling tasks with the attention mechanism. It does so using a novel attention mechanism called "multi-head attention". This technique first linearly transforms the input sample to three different learned representation, knows as the query (q), key (k), and value (v). Each of these representations is then split into multiple "heads", where each head "attends" to a different spatial segment of the representation independently and in parallel. A final output is produced by concatenating the outputs from all the heads and feeding this concatenation through a final linear transformation. This process is illustrated below (all images taken from Vaswani et al., 2017).

[[resources/multi-head-attention.png]]

Attention is useful because it can learn temporal correlations of arbitrary length very directly (i.e. without needing to be extracted through multiple convolutional layers or transmitted through multiple recurrent steps). Multi-head attention extends this advantage to the spatial domain while also allowing for parallelization. In the words of the authors:

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

### Encoder Layer
In sequence-to-sequence tasks, these attention layers are organized in an encoder-decoder architecture where the encoder and decoder each consist of stacked attention layers. The task of the encoder is to produce a representation of each sample in the sequence that the decoder can then use to produce a new sequence of different length, dimensionality, or both.

[[resources/transformer.png]]

However, as ours is a classification task, we omit the decoder and instead pass the encoded sequence through a small fully-connected network to produce a binary output. In addition, temporal position information is encoded into the input samples as the attention mechanism treats samples symmetrically with respect to time.

[[resources/transformer-clf.png]]

## Reinforced Attention
Zhang et al. (2019)<sup>[[2](https://ieeexplore.ieee.org/abstract/document/8970840)]</sup> used reinforcement learning to find a subset and ordering of input EEG channels that improves model performance. Since this technique learns to "attend" to a "zone" or subset of features, it could be considered a type of attention. And since it uses reinforcment learning (RL)—specifically dueling Q-networks (DQN)—to tune this "attention zone", it could be called "reinforced attention."

With this technique, a new feature set is constructed by "replicating and shuffling" the input features." Then, an RL task is set up as follows:
1. Some contiguous subsection of the new feature set is chosen as the initial "attention zone". The beginning and end of this attention zone comprise the states of the RL task.
2. A DQN agent is constructed to modify this attention zone. Specifically, its available actions are to either shift the attention zone to the left or right, expand it, or contract it.
3. A reward model is designed to evaluate the current state and provide a reward to the DQN agent. The reward can be constructed to balance raw performance with preference for a smaller attention zone.

Once the optimal attention zone has been learned, it is used to train the model used for the task. Zhang et al. use this technique in various EEG tasks—including activity classification, person identification, epilepsy diagnosis—as well as tasks involving multi-modal data (2018).<sup>[[3](https://www.ijcai.org/Proceedings/2018/0432.pdf)]</sup> They also use the technique with two different model types, CNN and LSTM.

The task model and the reward model do not have to be identical. In fact, Zhang et al. (2018)<sup>[[3](https://www.ijcai.org/Proceedings/2018/0432.pdf)]</sup> show that an approximate reward function can be used during the RL training phase to drastically reduce training time.

It's interesting to note that, though EEG data are time-series, this technique ignores any temporal statistics. Zhang et al. acknowledge that they are exploring methods to better exploit spatial correlations, not temporal, and each sample in their tasks is a single multi-channel reading.

## References
1. [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in neural information processing systems* (pp. 5998-6008).](http://papers.nips.cc/paper/7181-attention-is-all-you-need)
2. [Zhang, X., Yao, L., Wang, X., Zhang, W., Zhang, S., & Liu, Y. (2019, November). Know Your Mind: Adaptive Cognitive Activity Recognition with Reinforced CNN. In *2019 IEEE International Conference on Data Mining (ICDM)* (pp. 896-905). IEEE.](https://ieeexplore.ieee.org/abstract/document/8970840)
3. [Zhang, X., Yao, L., Huang, C., Wang, S., Tan, M., Long, G., & Wang, C. (2018). Multi-modality Sensor Data Classification with Selective Attention. *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence* (pp. 3111-3116)](https://www.ijcai.org/Proceedings/2018/0432.pdf)