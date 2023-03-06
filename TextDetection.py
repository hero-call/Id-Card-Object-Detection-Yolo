import numpy as np
import cv2
import glob

import BalanceAngle as ba




NET_CONFIG = "yolov3.cfg"
NET_WEIGHTS = "step2_6000.weights"
CLASSES = []
# BLOB_SIZE = (320, 320)
BLOB_SIZE = (416, 416)
CONFIDENT_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# WARNING = cv2.imread("warning.png")
# WARNING = cv2.resize(WARNING, BLOB_SIZE)


names_file = "classes.txt"
with open(names_file, "rt") as f:
    CLASSES = f.read().rstrip("\n").split("\n")


net = cv2.dnn.readNetFromDarknet(NET_CONFIG, NET_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)


def exit():
    if cv2.waitKey(1) == ord('0'):
        return 1
    return 0

def create_box(indices, img, details):
    bboxes, class_ids, confidences = details[0], details[1], details[2]
    for i in indices:
        bbox = bboxes[i]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, f'{CLASSES[class_ids[i]].upper()} {int(confidences[i]*100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        roi = img[y:y+h, x:x+w]
        name = "subImgs/"+CLASSES[class_ids[i]].upper()+".jpg"
        cv2.imwrite(name, roi)

def find_object(output, img):
    img_h, img_w, _ = img.shape
    bboxes = []
    class_ids = []
    confidences = []
    for member in output:
        for detect_vector in member:
            class_probabilities = detect_vector[5:] # have same index with classnames
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]
            if confidence > CONFIDENT_THRESHOLD:
                w = detect_vector[2] * img_w
                h = detect_vector[3] * img_h
                x = int((detect_vector[0] * img_w) - (w/2))
                y = int((detect_vector[1] * img_h) - (h/2))
                bboxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONFIDENT_THRESHOLD, NMS_THRESHOLD)
    details = [bboxes, class_ids, confidences]
    create_box(indices, img, details)

# test with camera
def test_camera():
    cap = cv2.VideoCapture(0)
    while not exit():
        success, frame = cap.read()

        angle = ba.find_angle(frame)
        if type(angle) == str:
            continue
        frame = ba.rotate(frame, angle)

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=BLOB_SIZE, mean=(0,0,0),
                                    swapRB=True, crop=False)
        # for img in blob:
        #     for k, b in enumerate(img):
        #         cv2.imshow(str(# test_camera()test_sik), b)

        net.setInput(blob)
        out_names = net.getUnconnectedOutLayersNames()
        output = net.forward(out_names)
        # print(type(output), output[0].shape, output[1].shape, output[2].shape)
        # print(output[0][0])
        find_object(output, frame)
        cv2.imshow("Webcam", frame)

# test with img
def test_single_img(img):
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=BLOB_SIZE, mean=(0,0,0),
                                    swapRB=True, crop=False)
    net.setInput(blob)
    out_names = net.getUnconnectedOutLayersNames()
    output = net.forward(out_names)
    find_object(output, img)
    cv2.imshow("Img", img)
    cv2.waitKey(0)





# test_camera()

# test_single_img(cv2.imread("01.jpeg"))


# imgs = glob.glob('*.jpeg')
# for img in imgs:
#     test_single_img(img)
