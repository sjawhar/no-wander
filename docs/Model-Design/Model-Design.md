## Definitions

We refer to the time surrounding a recovery as an "epoch". Each epoch is split into three "windows", which refer to the periods of time before and after a recovery and the label applied to segments that fall within that time. These windows are the distraction, uncertainty, and focus windows. Each window can be further broken down into many labeled "segments", which contain a sequence of samples from each data source. The label of each segment is determined by the window within which it falls.

Segments in the uncertainty window immediately preceding the recovery are discarded. From personal experience, this time period cannot be neatly categorized as either distraction or focus. Sometimes the distraction is noticed, but attention has not yet been returned to the object of focus. Sometimes attention returns in a piecemeal fashion. We believe we will have better results by not training on this time period.

[[resources/recovery-timeline.jpg]]

The above example illustrates a single epoch with a 4-second distraction window, 2-second focus window, 1-second uncertainty window, and 0.5 second segment size. These parameters result in 8 distraction segments and 4 focus segments for each epoch. At a recording frequency of 256Hz for EEG, each 0.5-second segment contains 128 x n samples: 128 samples from each of the n channels in the EEG stream.

This project explores the effectiveness of multiple model designs. Given the use case of this project (detecting distraction in a subject before the subject does), focus is given to architectures that support efficient, real-time inference. Please use the sidebar to the right to learn more about a particular design. **Please Note**: These models are still a work in progress, and details are subject to change.

## Methodology
To train and test our candidate models, we first extract all epochs, shuffle the order, then reserve 20% for model validation during development and 20% for final testing. The prepared datasets can be found as HDF5 files on [Amazon S3](https://no-wander-datasets.s3.amazonaws.com/). The "train" datafile contains two groups, "train" and "val", which respectively contain the training and validation epochs. The "test" datafile contains the testing datasets under the root group.

The software for this project is developed in Python, a standard for machine learning (ML). We also make use of ML frameworks such as Keras and TensorFlow as appropriate.

## Training Experiments
A history of ML model training experiments conducted for this project can be viewed in [this spreadsheet](https://docs.google.com/spreadsheets/d/1i03Li05xmXismmcXjPu_mv1lNQf7RambY2m5mYs2GpY/edit?usp=sharing). The first column in the sheet contains links to the experiments on Paperspace, which are publicly viewable. Some notes on navigating the Paperspace UI:
* The "Logs" tab contains the command used to create and train the model.
* Expanding the "Job Metrics (User-defined)" panel at the bottom of the "Metrics" tab will display charts showing the loss and accuracy of the model throughout training.
* The "Artifacts" tab contains an image of the model architecture as well as saved copies of the trained model.
* Some of the above features are missing for earlier jobs, as we were still improving our sharing protocols.

