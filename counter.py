import numpy as np
from scipy import spatial
import cv2


def box_previous_frames(previous_frame_detections, current_box, current_detections, frames_before_current):

    center_x, center_y, w, h = current_box
    dist = np.inf

    for i in range(frames_before_current):

        coordinate_list = list(previous_frame_detections[i].keys())
        if len(coordinate_list) == 0:
            continue

        temp_dist, index = spatial.KDTree(coordinate_list).query([(center_x, center_y)])
        if temp_dist < dist:
            dist = temp_dist
            frame_num = i
            coord = coordinate_list[index[0]]

    if dist > (max(w, h)/2):
        return False

    current_detections[(center_x, center_y)] = previous_frame_detections[frame_num][coord]
    return True


def count_obj(obj_detect, boxes, classes_ids, production, previous_frame_detections, frames_before_current):
    # creates the detection dictionary for the current frame
    current_detections = {}

    # checks if there are objects detected in the frame
    if len(obj_detect) > 0:

        for i in obj_detect.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            center_x = x + (w/2)
            center_y = y + (h/2)

            current_detections[(center_x, center_y)] = production

            if not box_previous_frames(previous_frame_detections, (center_x, center_y, w, h), current_detections, frames_before_current):
                production += 1

            # ID = current_detections.get((center_x, center_y, w, h))

            # if list(current_detections.values()).count(ID) > 1:

            current_detections[(center_x, center_y)] = production

    return production, current_detections
