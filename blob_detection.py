import cv2
import numpy as np

from blob_assignment import HungarianAlgorithm
from imutils.object_detection import non_max_suppression
from utils import \
    x1y1wh_to_x1y1x2y2_list, \
    x1y1x2y2_to_x1y1wh_list, \
    filter_blobs_by_distance, \
    filter_blobs_by_area


class BlobDetector:

    detector = None

    def __init__(self):
        self.threshold = [1, 100, 25]
        self.min_dist_between_blobs = 20
        self.filter_by_color = [True, 255]
        self.filter_by_area = [True, 35, 10000]
        self.filter_by_circularity = [False, 0.01, 1.0]
        self.filter_by_convexity = [False, 0.2, 1.0]
        self.filter_by_inertia = [False, 0, 1]
        self.detections = []
        self.blob_assigner = HungarianAlgorithm()

        # Setup SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = self.threshold[0]
        params.maxThreshold = self.threshold[1]
        params.thresholdStep = self.threshold[2]

        # Minimum distance between blobs
        params.minDistBetweenBlobs = self.min_dist_between_blobs

        # Filter by Color
        params.filterByColor = self.filter_by_color[0]
        params.blobColor = self.filter_by_color[1]

        # Filter by Area.
        params.filterByArea = self.filter_by_area[0]
        params.minArea = self.filter_by_area[1]
        params.maxArea = self.filter_by_area[2]

        # Filter by Circularity
        params.filterByCircularity = self.filter_by_circularity[0]
        params.minCircularity = self.filter_by_circularity[1]
        params.maxCircularity = self.filter_by_circularity[2]

        # Filter by Convexity
        params.filterByConvexity = self.filter_by_convexity[0]
        params.minConvexity = self.filter_by_convexity[1]
        params.maxConvexity = self.filter_by_convexity[2]

        # Filter by Inertia
        params.filterByInertia = self.filter_by_inertia[0]
        params.minInertiaRatio = self.filter_by_inertia[1]
        params.maxInertiaRatio = self.filter_by_inertia[2]

        self.detector = cv2.SimpleBlobDetector_create(params)

    def apply(self, image, frame_number):
        blobs = []

        im2 = np.copy(image)
        contours = cv2.findContours(im2,
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)[1]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            blobs.append((x, y, w, h))

        if blobs:
            if self.filter_by_area:
                blobs = filter_blobs_by_area(blobs, (self.filter_by_area[1],
                                                     self.filter_by_area[2]))
            blobs = filter_blobs_by_distance(blobs, self.min_dist_between_blobs)

            blobs = non_max_suppression(x1y1wh_to_x1y1x2y2_list(blobs),
                                        overlapThresh=0.25)
            blobs = x1y1x2y2_to_x1y1wh_list(blobs)

        self.detections = self.blob_assigner.apply(blobs,
                                                   self.detections,
                                                   frame_number)

        return blobs
