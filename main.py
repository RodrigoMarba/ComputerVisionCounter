import cv2
import numpy as np
import time
from counter import count_obj

start = time.time()

# loads model used in detection
net = cv2.dnn.readNetFromDarknet("yolov4_custom.cfg", "yolov4_custom_best.weights")

# reads the objects that the model was trained to detect and stores them in an array
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# defines the output layers to perform the detection
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# define and read the video
cap = cv2.VideoCapture("linha_cut.mp4")
ret, video = cap.read()  # reads the video

# initialize parameters
v_height, v_width, v_channels = video.shape  # defines height, width and channels of the video
fps = cap.get(cv2.CAP_PROP_FPS)  # defines video FPS
dt = float(fps / 60)  # defines the dt used in the Kalman filter
frames_before_current = 3  # frames that will be read to match characteristics

width = int(v_width * 60 / 100)
height = int(v_height * 60 / 100)
dim = (width, height)

boxes = []
confidences = []
classes_ids = []
threshold = 0.9
threshold_NMS = 0.2

production = 0  # starts counting at 0
previous_frame_detections = [{(0, 0): 0} for i in
                             range(
                                 frames_before_current)]  # creates the dictionary list that will store the detections (Xo, Yo, Lk, Hk)

# use when saving the video
# file_name = 'result_cut.avi'
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # MP4V
# out = cv2.VideoWriter(file_name, fourcc, fps, (v_width, v_height))

while cv2.waitKey(1) < 0:

    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    # resets reference for each detection:
    current_boxes = []
    boxes = []
    confidences = []
    classes_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > threshold:
                box = detection[0:4] * np.array([v_width, v_height, v_width, v_height])
                (center_x, center_y, w, h) = box.astype("int")

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                current_boxes.append([center_x, center_y, int(w), int(h)])
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                classes_ids.append(class_id)

    obj_detect = cv2.dnn.NMSBoxes(boxes, confidences, threshold, threshold_NMS)

    production, current_detections = count_obj(obj_detect, boxes, classes_ids, production, previous_frame_detections, frames_before_current)

    if len(obj_detect) > 0:
        for i in obj_detect.flatten():
            (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            cx = x + (w / 2)
            cy = y + (h / 2)
            cv2.rectangle(frame, (x, y,), (x + w, y + h), (255, 0, 255), 2)
            text_box = classes[classes_ids[i]] + ": " + str(round(confidences[i], 2))
            text_prod = "Contagem: " + str(production)
            print(text_prod)

            cv2.putText(frame, text_box, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 3)
            cv2.putText(frame, text_prod, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

    frame = cv2.resize(frame, dim)
    cv2.imshow("detection", frame)
    # out.write(frame)


    # excludes all detections of the frame used in the previous kalman
    # if there is no detection it can unexpected behavior
    previous_frame_detections.pop(0)

    # updates the detection list for use in the next kalman
    previous_frame_detections.append(current_detections)

end = time.time()
duration = end - start
print("vídeo processado em: " + str(duration))

print("end of script")
cv2.release()
cv2.destroyAllWindows()
