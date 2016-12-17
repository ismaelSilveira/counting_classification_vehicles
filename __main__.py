import numpy as np
import cv2

import background_subtraction as bg
import blob_detection as bd
from utils import \
    find_resolution_multiplier, \
    draw_blobs_and_line, \
    reduce_line


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
    print('Start to process images...')

    MAX_WIDTH = 320
    MAX_HEIGHT = 240

    # source = '/home/ismael/Desktop/ModTall/VideosTrafico/MVI_0022_xvid_001.avi'
    source = '/home/ismael/Desktop/ModTall/VideosTrafico/M6 Motorway Traffic.mp4'
    # source = '/home/ismael/Desktop/ModTall/VideosTrafico/UK Motorway M25 Trucks, Lorries, Cars Highway.mp4'

    cap = cv2.VideoCapture(source)
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

    while has_more_images:
        original = np.copy(frame)
        frame_number += 1

        # Line is defined
        if line_defined:
            frame_resized = cv2.resize(frame, (work_w, work_h))
            fgmask = background_subtractor.apply(frame_resized)
            # cv2.imshow("fgmask", fgmask)

            # Erode and Dilate the results for removing the noise
            morphed_mask = np.copy(fgmask)
            morphed_mask = cv2.morphologyEx(morphed_mask,
                                            cv2.MORPH_ERODE,
                                            cv2.getStructuringElement(
                                                cv2.MORPH_RECT,
                                                (3, 3)),
                                            iterations=1)
            morphed_mask = cv2.morphologyEx(morphed_mask,
                                            cv2.MORPH_DILATE,
                                            cv2.getStructuringElement(
                                                cv2.MORPH_RECT,
                                                (4, 4)),
                                            iterations=3)
            morphed_mask = cv2.morphologyEx(morphed_mask,
                                            cv2.MORPH_ERODE,
                                            cv2.getStructuringElement(
                                                cv2.MORPH_RECT,
                                                (2, 2)),
                                            iterations=12)
            morphed_mask = cv2.morphologyEx(morphed_mask,
                                            cv2.MORPH_DILATE,
                                            cv2.getStructuringElement(
                                                cv2.MORPH_RECT,
                                                (2, 6)),
                                            iterations=3)

            blobs = blob_detector.apply(morphed_mask, frame_number)

            cv2.imshow("Morphed", morphed_mask)

            color = (0, 255, 0)
            drawedBlobs = draw_blobs_and_line(original,
                                              blobs,
                                              line,
                                              color,
                                              resolution_multiplier,
                                              True)
            # cv2.imshow("Blobs", drawedBlobs)

            has_more_images, frame = cap.read()

        cv2.imshow("Original", original)

        k = cv2.waitKey(30) & 0xff
        if k in (ord('q'), ord('Q')):
            exit_cause = 'CLOSED BY PRESSING "Q|q"'
            break
        elif (k == ord('l')) and line_defined:
            print(line[0], line[1])

    cap.release()
    cv2.destroyAllWindows()