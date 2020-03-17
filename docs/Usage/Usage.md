## Requirements
* Muse 2
* Docker
* Linux
* Wireless keyboard

We're working on Windows and MacOS support.

## Commands
Each of the below commands can be run using Docker Compose by running the below command in the repository root.
```bash
docker-compose run --rm <command> [arguments]
```

Data will be stored to and/or read from the `data` directory.

### Record
Streams one or more of EEG, PPG, accelerometer, and gyroscope readings from the Muse 2. To signal a recovery, press one of the arrow keys on the keyboard. To end the recording early, press the "Escape" or "Q" button.

This project uses the [muse-lsl](https://github.com/alexandrebarachant/muse-lsl) library to interact with the Muse headband. Due to the perculiarities of muse-lsl or one of its dependencies, recording will hang if the connection to the headband is lost while recording. This would case all data for that session to be lost. As a safeguard against this kind of catastrophic data loss, this project "chunks" a session into five-minute incremements that are each saved to a separate file. This mechanism will cause small recording gaps between each chunk of around 15 seconds.

```bash
record [arguments] DURATION
```

**`DURATION`**  
Length of the meditation session, in minutes.

#### Optional Arguments
**`-a, --address STRING`**  
By default, a Bluetooth search is conducted to find your Muse, and the first found device is used. This behavior can be overridden by specifying a MAC address from which to stream.

**`-f, --filename STRING`**  
Filename prefix for recorded data. If you supply a value like `filename.csv`, chunks will be saved in files called `filename.n.SOURCE.csv`, where "n" is the chunk number and "SOURCE" is the one of "ACC", "EEG", "GYRO", or "PPG". By default, "filename" is the datetime of the recording in YYYY-mm-dd_HH-MM-SS format.

**`-s, --skip-visualize`**  
By default, a stability check is conducted after a connection is established. The streaming data is displayed and recording does not start until the signal stabilizes.

**`-t, --test`**  
By default, data is saved in the `data/input` directory. If this flag is provided, data is stored in the `data/test` directory instead.

**`-c, --acc`**  
Record accelerometer measurements

**`-g, --gyro`**  
Record gyroscope measurements

**`-p, --ppg`**  
Record PPG measurements

**`--no-eeg`**  
Don't record EEG measurements

### Process
Takes data files recorded using `record`, extracts recovery epochs, splits into train and test sets, and saves as datasets in h5 files. Each epoch includes the data 10 seconds before and after the recovery. If multiple data sources were recorded for a session chunk, they are combined in the saved dataset. Each dataset corresponds to an epoch and contains the following attributes:

* **chunk**: The number of the chunk in the session in which in this epoch was recorded
* **columns**: The names of the columns in the dataset
* **date**: The date of the session in which this epoch was recorded
* **recovery**: The index of the row at which the recovery was signalled
* **subject**: The ID of the subject to whom this recovery belongs

```bash
process [arguments] [DATA_DIR]
```

**`DATA_DIR`**  
Directory containing data files. Default is `data/input`

#### Optional Arguments
**`-l, --limit INT`**  
Limit the number of processed session chunks

**`-t, --test-split FLOAT`**  
Percentage of data to reserve for testing. Default is 0.2

**`-x, --aux-channel STRING`**
Channel name for Right Aux. Must be provided if Right Aux has data, otherwise channel is dropped.

### Train
Builds a model (currently only LSTM is supported) and saves the built model and diagram image in `MODEL_DIR`. If `epochs` is not 0, also trains the model on the data in `DATA_FILE` and saves the trained model and training history to `MODEL_DIR`. Even if `epochs` is not 0, `DATA_FILE` is required to determine the input size to the LSTM.

```bash
train [arguments] DATA_FILE MODEL_DIR
```

**`DATA_FILE`**  
Path to h5 file with labeled epochs.

**`MODEL_DIR`**  
Directory in which to save built model and images

#### Required Arguments
**`-s, --sample-size INT`**  
Number of readings/timesteps per sample

**`-q, --sequence-size INT`**  
Number of samples per LSTM sequence

**`-l, --lstm JSON`**  
JSON array of parameters to pass to LSTM(). `ic_params` controls IC layer after activation.

#### Optional Arguments
**`-c, --conv1d JSON`**  
JSON array of parameters to pass to Conv1D(). Default is to not use a conv layer. `ic_params` controls IC layer after activation. Include a `pool` attribute of kwargs to add a MaxPooling1D layer after IC layer.

**`-d, --dense JSON`**  
JSON object of parameters to pass to Dense(). Default is a 32-unit dense layer. `ic_params` controls IC layer after activation.

**`-p, --preprocess`**  
Type of preprocessing to perform on input data. Valid options are "extract-eeg", "normalize", and "none". Default is "none".

**`--dropout`**  
Dropout rate for input. Default is 0

**`--learning-rate FLOAT`**  
`learning_rate` parameter for optimizer. Default is 0.1

**`--beta-one FLOAT`**  
`beta_one` parameter for optimizer. Default is 0.9

**`--beta-two FLOAT`**  
`beta_two` parameter for optimizer. Default is 0.999

**`--decay FLOAT`**  
`decay` parameter for optimizer. Default is 0.01

**`--shuffle-samples`**  
Shuffle samples before constructing LSTM sequences

**`-t, --test-split FLOAT`**  
Ratio of data in data_file to use for validation. Default is 0

**`-e, --epochs INT`**  
Number of training epochs. Default is 1

**`-b, --batch-size INT`**  
Training batch size. Default is determined by `keras.Model.fit()`

**`-k, --checkpoint` / `--no-checkpoint`**  
Enabled / Disable saving of model checkpoint every epoch. Default is enabled

**`--tensorboard`**  
Save TensorBoard logs every epoch. Default is false

**`-g, --gradient-metrics`**  
Print metrics in Gradient chart format every epoch. Default is false
