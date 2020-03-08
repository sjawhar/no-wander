## Overview
Zhang et al. (2019)<sup>[[1](https://ieeexplore.ieee.org/abstract/document/8970840)]</sup> used reinforcement learning to find a subset and ordering of input EEG channels that improves model performance. Since this technique learns to "attend" to a "zone" or subset of features, it could be considered a type of attention. And since it uses reinforcment learning (RL)—specifically dueling Q-networks (DQN)—to tune this "attention zone", it could be called "reinforced attention."

With this technique, a new feature set is constructed by "replicating and shuffling" the input features." Then, an RL task is set up as follows:
1. Some contiguous subsection of the new feature set is chosen as the initial "attention zone". The beginning and end of this attention zone comprise the states of the RL task.
2. A DQN agent is constructed to modify this attention zone. Specifically, its available actions are to either shift the attention zone to the left or right, expand it, or contract it.
3. A reward model is designed to evaluate the current state and provide a reward to the DQN agent. The reward can be constructed to balance raw performance with preference for a smaller attention zone.

Once the optimal attention zone has been learned, it is used to train the model used for the task. Zhang et al. use this technique in various EEG tasks—including activity classification, person identification, epilepsy diagnosis—as well as tasks involving multi-modal data (2018).<sup>[[2](https://www.ijcai.org/Proceedings/2018/0432.pdf)]</sup> They also use the technique with two different model types, CNN and LSTM.

The task model and the reward model do not have to be identical. In fact, Zhang et al. (2018)<sup>[[2](https://www.ijcai.org/Proceedings/2018/0432.pdf)]</sup> show that an approximate reward function can be used during the RL training phase to drastically reduce training time.

It's interesting to note that, though EEG data are time-series, this technique ignores any temporal statistics. Zhang et al. acknowledge that they are exploring methods to better exploit spatial correlations, not temporal, and each sample in their tasks is a single multi-channel reading.

## References
1. [Zhang, X., Yao, L., Wang, X., Zhang, W., Zhang, S., & Liu, Y. (2019, November). Know Your Mind: Adaptive Cognitive Activity Recognition with Reinforced CNN. In *2019 IEEE International Conference on Data Mining (ICDM)* (pp. 896-905). IEEE.](https://ieeexplore.ieee.org/abstract/document/8970840)
2. [Zhang, X., Yao, L., Huang, C., Wang, S., Tan, M., Long, G., & Wang, C. (2018). Multi-modality Sensor Data Classification with Selective Attention. *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence* (pp. 3111-3116)](https://www.ijcai.org/Proceedings/2018/0432.pdf)