## Overview
As indicated by the title of the paper that introduced it ([Vaswani et al., 2017](http://papers.nips.cc/paper/7181-attention-is-all-you-need)), the Transformer model aims to replace all convolutional or recurrence operations in sequence modeling tasks with the attention mechanism. In sequence modeling tasks, attention is a technique for making more direct use of temporally-distant information by considering the entire input sequence at each timestep. In practice, this helps to capture long-range temporal correlations as well as resolve ambiguities (especially in language translation tasks).

## Multi-Head Attention
The Transformer model uses a novel attention mechanism called "multi-head attention". This technique first linearly transforms the input sample to three different learned representation, knows as the query (q), key (k), and value (v). Each of these representations is then split into multiple "heads", where each head "attends" to a different spatial segment of the representation independently and in parallel. A final output is produced by concatenating the outputs from all the heads and feeding this concatenation through a final linear transformation. This process is illustrated below (all images taken from Vaswani et al., 2017).

[[resources/multi-head-attention.png]]

Attention is useful because it can learn temporal correlations of arbitrary length very directly (i.e. without needing to be extracted through multiple convolutional layers or transmitted through multiple recurrent steps). Multi-head attention extends this advantage to the spatial domain while also allowing for parallelization. In the words of the authors:

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

## Architecture
In sequence-to-sequence tasks, these attention layers are organized in an encoder-decoder architecture where the encoder and decoder each consist of stacked attention layers. The task of the encoder is to produce a representation of each sample in the sequence that the decoder can then use to produce a new sequence of different length, dimensionality, or both.

[[resources/transformer.png]]

However, as ours is a classification task, we omit the decoder and instead pass the encoded sequence through a small fully-connected network to produce a binary output.

[[resources/transformer-clf.png]]

## References
1. [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In *Advances in neural information processing systems* (pp. 5998-6008).](http://papers.nips.cc/paper/7181-attention-is-all-you-need)