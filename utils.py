import cv2
from cv2 import putText
import numpy as np

import vehicle as vc

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


def x1y1wh_to_x1y1x2y2_list(rectangles):
    if len(rectangles) > 0:
        return np.apply_along_axis(x1y1wh_to_x1y1x2y2, 1, rectangles)
    else:
        return []


def x1y1x2y2_to_x1y1wh(rectangle):
    """
    Transform a rectangle expressed as (x1,y1,x2,y2) to (x1,y1, width, height)
    """
    return rectangle[0], rectangle[1], rectangle[2] - rectangle[0], \
        rectangle[3] - rectangle[1]


def x1y1x2y2_to_x1y1wh_list(rectangles):
    if len(rectangles) > 0:
        return np.apply_along_axis(x1y1x2y2_to_x1y1wh, 1, rectangles)
    else:
        return []


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


def draw_blobs_and_line(image, detections, line, color, count, bikes, cars,
                        trucks, cars_area, classify, classified,
                        vehicles_to_classify, multiplier=1, draw_center=False):
    color_line = (0, 0, 255)

    for detection in detections:
        blob_ = translate_blob(x1y1wh_to_x1y1x2y2(detection.position),
                               multiplier)

        cv2.rectangle(image,
                      (blob_[0], blob_[1]),
                      (blob_[2], blob_[3]),
                      color, 2)

        center_point = (int(blob_[0] + ((blob_[2] - blob_[0]) / 2)),
                        int(blob_[1] + ((blob_[3] - blob_[1]) / 2)))

        if draw_center:
            cv2.circle(image, center_point, 1, color, 2)

        if not detection.counted:
            if point_belongs_line(line, center_point, 6, (0, 20)):
                color_line = (255, 0, 0)
                count += 1
                detection.counted = True
                if classify:
                    # print("sin clasificar: ", detection.position, detection.position[2] * detection.position[3])
                    vehicles_to_classify.append(detection.copy())
                elif classified:
                    (bikes, cars, trucks) = classify_detection(detection,
                                                               cars_area,
                                                               bikes,
                                                               cars,
                                                               trucks)
                    # print(detection.vehicle, detection.position[2] * detection.position[3])
                    # print(bikes, cars, trucks)
        else:
            if not point_belongs_line(line, center_point, 6, (0, 20)):
                detection.counted = False

        if detection.counted:
            putText(image, "Si", (blob_[0], blob_[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    draw_selected_line(image, line, color_line)

    return count, detections, len(vehicles_to_classify) < 30, \
           vehicles_to_classify, bikes, cars, trucks


def resize_line(line, multiplier):
    return (tuple((int(x / multiplier) for x in line[0])),
            tuple((int(y / multiplier) for y in line[1])))


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


def point_belongs_line(line, point, threshold, extend=(0, 0)):
    inside = \
        ((line[0][0] - extend[0] <= point[0] <= line[1][0] + extend[0]) or
         (line[1][0] - extend[0] <= point[0] <= line[0][0] + extend[0])) and \
        ((line[0][1] - extend[1] <= point[1] <= line[1][1] + extend[1]) or
         (line[1][1] - extend[1] <= point[1] <= line[0][1] + extend[1]))

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
        return distance <= threshold
    else:
        return False


def calculate_cars_area(vehicles):
    values = [v.position[2] * v.position[3] for v in vehicles]

    return np.mean(values)


def classify_detection(detection, cars_area, bikes, cars, trucks):
    comparison_value = (cars_area * 0.3, cars_area * 3)
    detection_size = detection.position[2] * detection.position[3]
    if detection_size <= comparison_value[0]:
        detection.vehicle = vc.Vehicle.bike
        bikes += 1
    elif detection_size >= comparison_value[1]:
        detection.vehicle = vc.Vehicle.truck
        trucks += 1
    else:
        detection.vehicle = vc.Vehicle.car
        cars += 1

    return bikes, cars, trucks


def classify_vehicles(detections, cars_area):
    bikes = 0
    cars = 0
    trucks = 0

    comparison_value = (cars_area * 0.3, cars_area * 3)
    # print("comparison value=", comparison_value)
    for detection in detections:
        detection_size = detection.position[2] * detection.position[3]
        if detection_size <= comparison_value[0]:
            detection.vehicle = vc.Vehicle.bike
            bikes += 1
        elif detection_size >= comparison_value[1]:
            detection.vehicle = vc.Vehicle.truck
            trucks += 1
        else:
            detection.vehicle = vc.Vehicle.car
            cars += 1

    return bikes, cars, trucks


def filter_blobs_by_distance(blobs, distance):
    result = []

    for i, blob in enumerate(blobs):
        if blob in blobs:
            filtered_blobs = list(
                filter(lambda x: euclidean_distance(blob, x) < distance, blobs)
            )

            x_min = min(filtered_blobs, key=lambda x: x[0])[0]
            y_min = min(filtered_blobs, key=lambda x: x[1])[1]
            w_min = max(filtered_blobs, key=lambda x: x[0] + x[2])
            w_min = (w_min[0] + w_min[2]) - x_min
            h_min = max(filtered_blobs, key=lambda x: x[1] + x[3])
            h_min = (h_min[1] + h_min[3]) - y_min
            result.append((x_min, y_min, w_min, h_min))
            for filtered_blob in filtered_blobs:
                blobs = [x for x in blobs if x != filtered_blob]

    return result


def filter_blobs_by_area(blobs, area):
    return list(filter(lambda x: area[0] <= (x[2] * x[3]) <= area[1], blobs))
