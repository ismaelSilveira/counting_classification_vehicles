import numpy as np
import cv2

from config import config


class BackgroundSubtractor:

    subtractor = None

    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2()

        self.subtractor.setHistory(config.getint('HISTORY'))
        self.subtractor.setDetectShadows(config.getboolean('DETECT_SHADOWS'))
        self.subtractor.setShadowThreshold(config.getfloat('SHADOW_THRESHOLD'))
        self.subtractor.setShadowValue(0)
        self.subtractor.setNMixtures(config.getint('N_MIXTURES'))

    def apply(self, frame):

        blurred_frame = np.copy(frame)
        blurred_frame = cv2.morphologyEx(blurred_frame,
                                         cv2.MORPH_DILATE,
                                         cv2.getStructuringElement(
                                             cv2.MORPH_RECT,
                                             (2, 2)),
                                         iterations=1)

        # Aplico la deteccion de fondo, esto tiene en cuenta el o los frames
        # previamente cargados
        fgmaskknn = self.subtractor.apply(blurred_frame)

        return fgmaskknn
