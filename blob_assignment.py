import numpy as np
import math
from munkres import Munkres

from detection import Detection
from utils import blob_center, euclidean_distance, INFINITE


def get_costs_matrix(actual_blobs, detections):
    # the costs matrix width has to be larger or equal than height
    rows_count = len(actual_blobs)

    if rows_count > len(detections):
        columns_count = rows_count
        for i in range(0, len(actual_blobs) - len(detections)):
            detections = np.append(detections, Detection())
    else:
        columns_count = len(detections)

    costs_matrix = np.zeros(shape=(rows_count, columns_count), dtype=float)

    for i, blob in enumerate(actual_blobs):
        for j, detection in enumerate(detections):
            if detection.position is None:
                costs_matrix[i][j] = INFINITE
            else:
                costs_matrix[i][j] = euclidean_distance(
                    blob_center(blob), blob_center(detection.position)
                )

    return costs_matrix, detections


def blob2detection(blob, frame_number):
    return Detection(blob, frame_number)


class HungarianAlgorithm:
    def __init__(self):
        self.munkres_ = Munkres()

    def apply(self, actual_blobs, detections, frame_number):
        costs, detections = get_costs_matrix(actual_blobs, detections)
        to_delete = []

        if len(costs) > 0:
            indexes = self.munkres_.compute(np.absolute(costs))
            for index in indexes:
                if detections[index[1]].position is None:
                    detections[index[1]].frame_detected = frame_number
                detections[index[1]].position = actual_blobs[index[0]]
                detections[index[1]].last_update = frame_number
                detections[index[1]].size = (actual_blobs[index[0]][2],
                                             actual_blobs[index[0]][3])

            if len(detections) > len(indexes):
                for i in range(0, len(detections)-1):
                    if (i not in [ind[1] for ind in indexes]) and \
                            ((frame_number - detections[i].last_update) > 5):
                        to_delete.append(i)
        else:
            for i in range(0, len(detections)):
                if (frame_number - detections[i].last_update) > 5:
                    to_delete.append(i)

        detections = np.delete(detections, to_delete)

        return detections
