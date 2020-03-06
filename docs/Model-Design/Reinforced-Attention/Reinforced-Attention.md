## Overview
* Uses individual samples instead of sequences
* Leverages spatial relationships instead of temporal
* Features are replicated and shuffled. Reinforcement learning is used to learn the best subsection and ordering of features.
* Reward model can be task-specific (CNN+KNN and WAS-LSTM are explored)
* Approximate reward function can be used to speed up training

## References
1. [Zhang, X., Yao, L., Wang, X., Zhang, W., Zhang, S., & Liu, Y. (2019, November). Know Your Mind: Adaptive Cognitive Activity Recognition with Reinforced CNN. In *2019 IEEE International Conference on Data Mining (ICDM)* (pp. 896-905). IEEE.](https://ieeexplore.ieee.org/abstract/document/8970840)
2. [Zhang, X., Yao, L., Huang, C., Wang, S., Tan, M., Long, G., & Wang, C. (2018). Multi-modality Sensor Data Classification with Selective Attention. *Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence* (pp. 3111-3116)](https://www.ijcai.org/Proceedings/2018/0432.pdf)