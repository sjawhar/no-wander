import logging
from .constants import PACKAGE_NAME
from muselsl.constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK
from muselsl.viewer_v2 import Canvas as MuseCanvas
from pylsl import resolve_byprop, StreamInlet
from vispy import app


SIGNAL_QUALITY_THRESHOLD = 10
SIGNAL_STABILITY_COUNT = 100

logger = logging.getLogger(PACKAGE_NAME + '.' + __name__)


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
