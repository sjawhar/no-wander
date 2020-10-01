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

**`-p, --probes MEAN [STD]`**  
Sample user focus with audio probes. Provide one number X to sample every X minutes. Provide two numbers MEAN, STD to sample every Gaussian(MEAN, STD) minutes. Default is not to use probes.

**`-s, --subject-id STRING`**  
Unique identifier for the current subject, to be included in session info.

**`-no-visualize`**  
By default, a stability check is conducted after a connection is established. The streaming data is displayed and recording does not start until the signal stabilizes.

**`-t, --test`**  
By default, data is saved in the `data/input` directory. If this flag is provided, data is stored in the `data/test` directory instead.

**`-c, --acc`**  
Record accelerometer measurements

**`-g, --gyro`**  
Record gyroscope measurements

**`--ppg`**  
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

**`-s, --val-split FLOAT`**  
Percentage of data to reserve for validation. Default is 0.2

**`-t, --test-split FLOAT`**  
Percentage of data to reserve for final testing. Default is 0.2

**`-x, --aux-channel STRING`**
Channel name for Right Aux. Must be provided if Right Aux has data, otherwise channel is dropped.

### Train
Builds a model (currently only LSTM is supported) and saves the built model and diagram image in `MODEL_DIR`. If `--epochs` is not 0, also trains the model on the data in `DATA_FILE` and saves the trained model and training history to `MODEL_DIR`. Even if `--epochs` is not 0, `DATA_FILE` is required to determine the input size to the LSTM.

```bash
train [arguments] DATA_FILE MODEL_DIR
```

**`DATA_FILE`**  
Path to h5 file with labeled epochs.

**`MODEL_DIR`**  
Directory in which to save built model and images

#### Required Arguments
**`-s, --segment-size INT`**  
Number of samples/readings/timesteps per segment

**`-q, --sequence-size INT`**  
Number of segments per LSTM sequence

**`-l, --layers JSON`**  
JSON array of layers with params. `type` controls layer type. `ic_params` controls IC layer after activation. Include a `pool` attribute of kwargs in a Conv1D layer to add a MaxPooling1D layer before the IC layer. A single layer controlled by `--output` is automatically added after all specified layers.

#### Optional Arguments
**`--o, --output JSON`**
JSON object overriding default output layer specification. Default is a single-unit Dense layer with sigmoid activation. Do **NOT** include `type` or `name` in the layer specification.

**`--pre-window FLOAT FLOAT`**  
Start and end of pre-recovery window, in seconds. Should be negative numbers. Default is `-7 -1`

**`--post-window FLOAT FLOAT`**  
Start and end of post-recovery window, in seconds. Default is `0 3`

**`-p, --preprocess`**  
Type of preprocessing to perform on input data. Valid options are "extract-eeg", "normalize", and "none". Default is "none".

**`--encode-position`**
Add positional encoding to input, before dropout. Default is false

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

**`--shuffle-segments`**  
Shuffle segments before constructing LSTM sequences

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

#### Specifying Layers
`--layers` should be list of objects in JSON format, where each object in the list contains the specification of a single layer. The `type` key in each object must correspond to the name of a Layer class in `tensorflow.keras.layers`. To create only one layer, you can also use a single JSON object instead of an array of length 1. Please refer to the YAML files in the .ps_project folder for examples.

In general layers use their default values as specified in the Keras/TensorFlow documentation, with the following exceptions:
* `Dense` and `Conv1D` layers default to `activation="relu"`.
* An `LSTM` layer defaults to `return_sequences=False` if it is the final LSTM layer and `return_sequences=True` otherwise.

Using `Encoder` as the layer type will create a multi-head attention (MHA) + point-wise feed-forward (FF) layer stack, in the style of the Transformer network. This layer type accepts the following parameters:
* `attention_activation`: Activation to use inside MHA layer. Default is `null`
* `dropout`: Dropout to use after MHA and FF layers, before residual connection. Default is `0.1`
* `epsilon`: Divide-by-zero check parameter for [LayerNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization). Default is `1e-6`
* `ff_activation`: Activation to use on hidden Dense layer of FF layer. Default is `"relu"`
* `ff_units`: Units of hidden Dense layer of FF layer. Default is `1000`
* `num_heads`: Number of MHA heads. Default is `8`

Each layer specification object can include one or more regularization parameters (e.g. `kernel_regularizer`). The value of these parameters should be either a string or a dict:
  * `"l1"` or `"l2"` will create the corresponding regularizer with default parameters.
  * `{"l1": 0.1, "l2": 0.2}` will create an `L1L2` regularizer with parameters `l1=0.1, l2=0.2`. You don't need to include both l1 and l2 values.

#### Other Notes
* If `--shuffle-segments` is not true, only segments belonging to contiguous sequences of length `--sequence-size` are used.
* If you include more flags in your command that are not listed above, they will be passed as kwargs to `model.fit()`.
