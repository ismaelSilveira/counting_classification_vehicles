import numpy as np
import cv2

import background_subtraction as bg
import blob_detection as bd
from utils import \
    find_resolution_multiplier, \
    draw_blobs_and_line, \
    reduce_line, \
    draw_selected_line, \
    calculate_cars_area, \
    classify_vehicles
from config import config


# mouse callback function
def define_line(event, x, y, flags, param):
    global line, line_resized, line_defined

    if event == cv2.EVENT_LBUTTONUP:
        if len(line) < 2:
            line.append((x, y))
            if len(line) == 2:
                line_defined = True
                print("Line: ", line)
                line_resized = reduce_line(line, param[0])

if __name__ == '__main__':

    debug = config.getboolean('DEBUG')
    show_binary_image = config.getboolean('SHOW_BINARY_IMAGE')
    source = config.get('SOURCE')
    history = config.getint('HISTORY')

    print('Start to process images...')
    cap = cv2.VideoCapture(source)

    # Original FPS
    try:
        FPS = float(int(cap.get(cv2.CAP_PROP_FPS)))
        if FPS == 0.:
            FPS = 30
    except ValueError:
        FPS = 30

    print("Working at", FPS, "FPS")

    # Getting width and height of captured images
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Real resolution: Width", w, "Height", h)
    resolution_multiplier = find_resolution_multiplier(w, h)
    work_w = int(w / resolution_multiplier)
    work_h = int(h / resolution_multiplier)
    print("Work resolution: Width", work_w, "Height", work_h)

    background_subtractor = bg.BackgroundSubtractor()
    blob_detector = bd.BlobDetector()

    has_more_images = True
    line = []
    line_resized = []
    line_defined = False

    cv2.namedWindow("Original")
    cv2.setMouseCallback("Original",
                         define_line,
                         param=[
                             resolution_multiplier
                         ])

    has_more_images, frame = cap.read()
    frame_number = 1
    count = 0

    classify = True
    classified = False
    vehicles_to_classify = []
    bikes = 0
    cars = 0
    trucks = 0
    cars_area = 0

    while has_more_images:
        original = np.copy(frame)
        frame_number += 1

        # Line is defined
        if line_defined:
            frame_resized = cv2.resize(frame, (work_w, work_h))
            fgmask = background_subtractor.apply(frame_resized)

            if frame_number > history:
                # Erode and Dilate the results for removing the noise
                morphed_mask = np.copy(fgmask)
                morphed_mask = cv2.morphologyEx(morphed_mask,
                                                cv2.MORPH_CLOSE,
                                                cv2.getStructuringElement(
                                                    cv2.MORPH_RECT, (5, 5)),
                                                iterations=2)
                morphed_mask = cv2.morphologyEx(morphed_mask,
                                                cv2.MORPH_OPEN,
                                                cv2.getStructuringElement(
                                                    cv2.MORPH_RECT, (3, 3)),
                                                iterations=2)
                morphed_mask = cv2.morphologyEx(morphed_mask,
                                                cv2.MORPH_CLOSE,
                                                cv2.getStructuringElement(
                                                    cv2.MORPH_RECT, (5, 5)),
                                                iterations=1)
                morphed_mask = cv2.morphologyEx(morphed_mask,
                                                cv2.MORPH_ERODE,
                                                cv2.getStructuringElement(
                                                    cv2.MORPH_RECT, (5, 5)),
                                                iterations=1)

                blobs = blob_detector.apply(morphed_mask, frame_number)

                if show_binary_image:
                    cv2.imshow("Morphed", morphed_mask)

                color = (0, 255, 0)
                (count, blob_detector.detections,
                 classify, vehicles_to_classify,
                 bikes, cars, trucks) = \
                    draw_blobs_and_line(original,
                                        blob_detector.detections,
                                        line,
                                        color,
                                        count,
                                        bikes,
                                        cars,
                                        trucks,
                                        cars_area,
                                        classify,
                                        classified,
                                        vehicles_to_classify,
                                        resolution_multiplier,
                                        True)

                if not (classify or classified):
                    cars_area = calculate_cars_area(vehicles_to_classify)
                    print("Cars area=", cars_area)
                    classified = True
                    (bikes, cars, trucks) = \
                        classify_vehicles(vehicles_to_classify, cars_area)
                    if debug:
                        print(bikes, cars, trucks)
            else:
                draw_selected_line(original, line, (0, 0, 255))

            has_more_images, frame = cap.read()

        cv2.imshow("Original", original)

        k = cv2.waitKey(1) & 0xff
        if k in (ord('q'), ord('Q')):
            exit_cause = 'CLOSED BY PRESSING "Q|q"'
            break
        elif (k == ord('l')) and line_defined:
            print(line[0], line[1])

    print("Total count=", count)
    print("Total bikes=", bikes)
    print("Total cars=", cars)
    print("Total trucks=", trucks)

    cap.release()
    cv2.destroyAllWindows()
