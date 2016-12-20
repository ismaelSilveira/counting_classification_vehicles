import cv2
import numpy as np

MAX_WIDTH = 320
MAX_HEIGHT = 240
INFINITE = 100000000000


def find_resolution_multiplier(w, h):
    """
    Find a resolution divisor to get manageable resolution and then get the
    original back to show.
    :param w: real width
    :param h: real height
    :return: a float number
    """
    if w > MAX_WIDTH or h > MAX_HEIGHT:
        mult_w = w / MAX_WIDTH
        mult_h = h / MAX_HEIGHT
        if mult_w > mult_h:
            return mult_w
        else:
            return mult_h
    else:
        return 1


def x1y1wh_to_x1y1x2y2(rectangle):
    """
    Transform a rectangle expressed as (x1,y1, width, height) to (x1,y1,x2,y2)
    """
    return (rectangle[0],
            rectangle[1],
            rectangle[0] + rectangle[2],
            rectangle[1] + rectangle[3])


def reduce_line(line, multiplier=1):
    return [(int(line[0][0] / multiplier), int(line[0][1] / multiplier)),
            (int(line[1][0] / multiplier), int(line[1][1] / multiplier))]


def translate_line(line, multiplier=1):
    return [(int(line[0][0] * multiplier), int(line[0][1] * multiplier)),
            (int(line[1][0] * multiplier), int(line[1][1] * multiplier))]


def translate_blob(blob, multiplier=1):
    return int(blob[0] * multiplier), int(blob[1] * multiplier), \
           int(blob[2] * multiplier), int(blob[3] * multiplier)


def draw_selected_line(image, line, color, multiplier=1):
    line = translate_line(line, multiplier)
    return cv2.line(image, line[0], line[1], color, 2)


def draw_blobs_and_line(image, blobs, line, color, multiplier=1, draw_center=False):
    color_line = (0, 0, 255)

    for blob in blobs:
        blob_ = translate_blob(x1y1wh_to_x1y1x2y2(blob), multiplier)

        cv2.rectangle(image,
                      (blob_[0], blob_[1]),
                      (blob_[2], blob_[3]),
                      color, 2)

        center_point = (int(blob_[0] + ((blob_[2] - blob_[0]) / 2)),
                        int(blob_[1] + ((blob_[3] - blob_[1]) / 2)))

        if draw_center:
            cv2.circle(image, center_point, 1, color, 2)

        if point_belongs_line(line, center_point, 1, multiplier):
            color_line = (255, 0, 0)

    draw_selected_line(image, line, color_line)


def resize_line(line, multiplier):
    return (tuple((int(x / multiplier) for x in line[0])),
            tuple((int(y / multiplier) for y in line[1])))


def point_belongs_line(line, point, threshold, multiplier=1):
    inside = ((line[0][0] <= point[0] <= line[1][0]) or
              (line[1][0] <= point[0] <= line[0][0])) and \
             ((line[0][1] <= point[1] <= line[1][1]) or
              (line[1][1] <= point[1] <= line[0][1]))

    if inside:
        if line[1][1] == line[0][1]:
            distance = abs(point[1] - line[0][1])
        elif line[1][0] == line[0][0]:
            distance = abs(point[0] - line[0][0])
        else:
            distance = abs(
                ((point[0] - line[0][0]) / (line[1][0] - line[0][0])) -
                ((point[1] - line[0][1]) / (line[1][1] - line[0][1]))
            )

        return distance < (threshold / multiplier)
    else:
        return False


def blob_center(blob):
    return int(blob[0] + (blob[2]/2)), int(blob[1] + (blob[3]/2))


def euclidean_distance(point1, point2):
    """
    Returns the euclidean distance between point 1 and 2.
    :param point1: Tuple with the position of the point 1
    :param point2: Tuple with the position of the point 2
    :return:
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))
