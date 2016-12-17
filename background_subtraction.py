import numpy as np
import cv2


class BackgroundSubtractor:

    subtractor = None

    def __init__(self):

        self.subtractor = cv2.createBackgroundSubtractorKNN()

        # Configuration parameters
        self.gaussian_size = (5, 5)
        self.erode_frame_size = np.ones((5, 5), np.uint8)
        self.erode_frame_times = 2
        self.dilate_frame_size = np.ones((5, 5), np.uint8)
        self.dilate_frame_times = 1

        self.history = 15
        self.dist_2_threshold = 200
        self.n_samples = 10
        self.knn_samples = 5
        self.detect_shadows = True
        self.shadow_threshold = 0.7
        self.shadow_value = 0

        # Sets the number of last frames that affect the background model.
        self.subtractor.setHistory(self.history)

        # Sets the threshold on the squared distance between the pixel and
        # the sample. The threshold on the squared distance between the
        # pixel and the sample to decide \
        # whether a pixel is close to a data sample.
        self.subtractor.setDist2Threshold(self.dist_2_threshold)

        # Sets the shadow detection flag. \
        # If true, the algorithm detects shadows and marks them.
        self.subtractor.setDetectShadows(self.detect_shadows)

        # Sets the number of neighbours, the k in kNN. \
        # K is the number of samples that need to be within dist2Threshold
        # in order \
        # to decide that that pixel is matching the kNN background model.
        # Sets the k in the kNN. How many nearest neighbors need to match.
        self.subtractor.setkNNSamples(self.knn_samples)

        # Sets the shadow threshold. A shadow is detected if pixel is a
        # darker version of the background. The shadow threshold
        # (Tau in the paper) is a threshold defining how much darker the
        # shadow can be. Tau= 0.5 means that if a pixel is more than twice \
        # darker then it is not shadow. See Prati, Mikic,
        # Trivedi and Cucchiarra, \
        # Detecting Moving Shadows...*, IEEE PAMI,2003.
        self.subtractor.setShadowThreshold(self.shadow_threshold)

        # Sets the number of data samples in the background model. \
        # The model needs to be reinitialized to reserve memory.
        self.subtractor.setNSamples(self.n_samples)

        # Sets the value used to mark shadows in the foreground mask
        self.subtractor.setShadowValue(self.shadow_value)

    def apply(self, frame):

        # Convierto imagen a escalas de grises
        # bg = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        # Aplico filtro Blur
        blurred_frame = np.copy(frame)
        # blurred_frame = cv2.GaussianBlur(blurred_frame,
        #                                  (5, 5),
        #                                  10)

        blurred_frame = cv2.morphologyEx(blurred_frame,
                                         cv2.MORPH_ERODE,
                                         cv2.getStructuringElement(
                                             cv2.MORPH_RECT,
                                             (5, 5)),
                                         iterations=2)

        blurred_frame = cv2.morphologyEx(blurred_frame,
                                         cv2.MORPH_DILATE,
                                         cv2.getStructuringElement(
                                             cv2.MORPH_RECT,
                                             (5, 5)),
                                         iterations=1)

        # Aplico la deteccion de fondo, esto tiene en cuenta el o los frames
        # previamente cargados
        fgmaskknn = self.subtractor.apply(blurred_frame)

        return fgmaskknn
