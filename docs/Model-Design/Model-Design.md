## Definitions

We refer to the time surrounding a recovery as an "epoch". Each epoch is split into three "windows", which refer to the segments of time before and after a recovery and the label applied to samples that fall within that time. These windows are the distraction, uncertainty, and focus windows. Each window can be further broken down into many labeled "samples", which contain a sequence of measurements from each data source. The label of each sample is determined by the window within which it falls.

Samples in the uncertainty window immediately preceding the recovery are discarded. From personal experience, this time period cannot be neatly categorized as either distraction or focus. Sometimes the distraction is noticed, but attention has not yet been returned to the object of focus. Sometimes attention returns in a piecemeal fashion. We believe we will have better results by not training on this time period.

[[resources/recovery-timeline.jpg]]

The above example illustrates a single epoch with a 4-second distraction window, 2-second focus window, 1-second uncertainty window, and 0.5 second sample size. These parameters result in 8 distraction samples and 4 focus samples for each epoch. At a recording frequency of 256Hz for EEG, each 0.5-second sample contains 128 x n datapoints: 128 datapoints from each of the n channels in the EEG stream.

This project will explore the effectiveness of multiple model designs. Please use the sidebar to the right to learn more about a particular design. **Please Note**: These models are still a work in progress, and details are subject to change.

## Methodology
TODO
* Train/test split + Validation holdout
* Stratified cross-validation
* **Realtime prediction**
* Tools (Python, Keras, TensorFlow)