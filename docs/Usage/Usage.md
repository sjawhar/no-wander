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

#### Required Arguments
**`-d, --duration DURATION`**  
Length of the meditation session, in minutes.

#### Optional Arguments
**`-a, --address ADDRESS`**  
By default, a Bluetooth search is conducted to find your Muse, and the first found device is used. This behavior can be overridden by specifying a MAC address from which to stream.

**`-f, --filename FILENAME`**  
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
TODO