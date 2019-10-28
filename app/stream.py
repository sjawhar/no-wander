import logging
from multiprocessing import Process
from muselsl import stream
from muselsl.constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK
from muselsl.viewer_v2 import Canvas as MuseCanvas
from pylsl import resolve_byprop, StreamInlet
from vispy import app
from .constants import PACKAGE_NAME


ATTEMPTS_MAX = 5
SIGNAL_QUALITY_THRESHOLD = 10
SIGNAL_STABILITY_COUNT = 100
SOURCE_ACC = 'ACC'
SOURCE_EEG = 'EEG'
SOURCE_GYRO = 'GYRO'
SOURCE_PPG = 'PPG'

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


def start_stream(address, sources, confirm=True):
    if address is None:
        logger.info('No address provided, searching for muses...')
        muses = list_muses()
        if not muses or len(muses) == 0:
            logger.error('No muses found, quitting.')
            exit(1)
        address = muses[0]['address']
        logger.debug(f'Found muse with address {address}.')

    kwargs = {
        'acc_enabled': SOURCE_ACC in sources,
        'eeg_disabled': SOURCE_EEG not in sources,
        'gyro_enabled': SOURCE_GYRO in sources,
        'ppg_enabled': SOURCE_PPG in sources,
    }
    keep_trying = True
    while keep_trying:
        logger.info(f'Establishing stream to {address}...')
        attempt = 0
        while attempt < ATTEMPTS_MAX:
            attempt += 1
            logger.info(f'Beginning stream attempt {attempt}...')
            stream_process = Process(target=stream, args=(address,), kwargs=kwargs)
            stream_process.daemon = True
            stream_process.start()
            stream_process.join(7)
            if not stream_process.is_alive():
                logger.warning(f'Streaming attempt {attempt} failed.')
                continue

            def stream_manager(restart=True):
                logger.debug(f'Stream process alive: {stream_process.is_alive()}')
                stream_process.terminate()
                return start_stream(address, sources, False) if restart else False

            logger.info('Stream established!')
            return stream_manager

        logger.error(f'Could not establish stream after {attempt} attempts.')
        if not confirm:
            continue
        keep_trying = str.lower(input('Keep trying? (y/N) ')) in ['y', 'yes']

    return False

def visualize():
    logger.info("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))

    logger.debug("Start acquiring data.")
    inlet = StreamInlet(streams[0], max_chunklen=LSL_EEG_CHUNK)
    Canvas(inlet)
    app.run()


class Canvas(MuseCanvas):
    def __init__(self, lsl_inlet, scale=500, filt=True):
        self.quality_count = 0
        super().__init__(lsl_inlet, scale, filt)


    def on_draw(self, event):
        max_quality = max(float(q.text) for q in self.quality)
        logger.debug('Signal quality is %.2f' % max_quality)
        if max_quality > SIGNAL_QUALITY_THRESHOLD:
            logger.debug('Signal quality poor, counter reset')
            self.quality_count = 0
        else:
            self.quality_count += 1
            logger.debug(f'Signal quality counter at {self.quality_count}')

        if self.quality_count >= SIGNAL_STABILITY_COUNT:
            logger.info('Signal stabilized!')
            return app.quit()

        super().on_draw(event)
