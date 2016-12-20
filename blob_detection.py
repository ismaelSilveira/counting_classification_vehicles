import cv2

from blob_assignment import HungarianAlgorithm
from detection import Detection


class BlobDetector:

    detector = None

    def __init__(self):
        self.threshold = [1, 100, 50]
        self.min_dist_between_blobs = 20
        self.filter_by_color = [True, 255]
        self.filter_by_area = [True, 50, 100000]
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

        for keyPoint in self.detector.detect(image):
            x1 = max(int(keyPoint.pt[0] - (keyPoint.size / 2)), 0)
            y1 = max(int(keyPoint.pt[1] - (keyPoint.size / 2)), 0)
            x2 = min(int(keyPoint.pt[0] + (keyPoint.size / 2)), image.shape[1])
            y2 = min(int(keyPoint.pt[1] + (keyPoint.size / 2)), image.shape[0])

            blobs.append((x1, y1, x2 - x1, y2 - y1))

        self.detections = self.blob_assigner.apply(blobs, self.detections, frame_number)

        return blobs
