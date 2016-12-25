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

    def __init__(self):
        self.min_dist_between_blobs = 25
        self.filter_by_area = [True, 30, 10000]
        self.detections = []
        self.blob_assigner = HungarianAlgorithm()

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
                                        overlapThresh=0.2)
            blobs = x1y1x2y2_to_x1y1wh_list(blobs)

        self.detections = self.blob_assigner.apply(blobs,
                                                   self.detections,
                                                   frame_number)

        return blobs
