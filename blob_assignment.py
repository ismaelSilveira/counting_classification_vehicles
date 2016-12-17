import numpy as np
import math
from munkres import Munkres

from utils import blob_center, euclidean_distance


def get_costs_matrix(actual_blobs, detections):

    # the costs matrix width has to be larger or equal than height
    columns_count = len(actual_blobs) \
        if len(detections) < len(actual_blobs) else len(detections)
    rows_count = len(actual_blobs)

    costs_matrix = np.zeros(shape=(rows_count, columns_count), dtype=float)
    print(costs_matrix, costs_matrix.shape)

    # for i in range(0, rows_count):
    #     costs_row = np.zeros(shape=columns_count, dtype=float)
    #
    #     for j in range(0, columns_count):
    #         costs_row[j] = euclidean_distance(blob_center(actual_blobs[i]),
    #                                           blob_center(detections[j].vehicle))
    #     costs_matrix[i] = costs_row

    # columns_relation = []
    # columns_to_delete = []
    # rows_to_delete = []
    # j = 0
    # for i in range(0, len(assigned_column)):
    #     if assigned_column[i] == -1:
    #         columns_to_delete.append(i)
    #     else:
    #         # column j corresponds to original column_data i
    #         columns_relation.append((j, i))
    #         j += 1
    # for i in range(0, len(assigned_row)):
    #     if assigned_row[i] == -1:
    #         rows_to_delete.append(i)
    #
    # costs_matrix = np.delete(costs_matrix, columns_to_delete, axis=1)
    # costs_matrix = np.delete(costs_matrix, rows_to_delete, axis=0)
    #
    # valid_columns_amount = costs_matrix.shape[1]
    # new_columns = costs_matrix.shape[0] - costs_matrix.shape[1]
    # if new_columns > 0:
    #     # more rows than columns
    #     a = np.empty((costs_matrix.shape[0], new_columns))
    #     a.fill(-self.infinite)
    #     costs_matrix = np.append(costs_matrix, a, axis=1)

    return costs_matrix


class HungarianAlgorithm:
    def __init__(self):
        self.munkres_ = Munkres()

    def apply(self, actual_blobs, detections):
        assigned_row = []
        assigned_row_cost = np.empty(shape=len(actual_blobs), dtype=float)
        assigned_row_cost.fill(-1.0)

        if len(actual_blobs) > 0:
            if len(detections) > 0:

                costs = get_costs_matrix(actual_blobs, detections)
                print(costs)

            #     if costs.shape[0] > 0 and costs.shape[1]:
            #         indexes = self.munkres_.compute(np.absolute(costs))
            #
            #         j = 0
            #         for i in range(0, len(assigned_row)):
            #             if assigned_row[i] == 0:
            #                 column = indexes[j][1]
            #
            #                 if column < valid_columns_amount and \
            #                         math.copysign(1, costs[j][column]) == 1.0:
            #                     assigned_row[i] = columns_relation[column][1]
            #                 else:
            #                     assigned_row[i] = -1
            #                 assigned_row_cost[i] = abs(costs[j][column])
            #
            #                 j += 1
            # else:
            #     assigned_row = np.empty(shape=len(rows_data), dtype=int)
            #     assigned_row.fill(-1)

        return assigned_row, assigned_row_cost
