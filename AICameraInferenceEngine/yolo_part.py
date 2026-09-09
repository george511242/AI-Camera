import cv2

CONF_THRESHOLD = 0.7
NET = cv2.dnn.readNetFromDarknet(
    r"/model/person/yolo.cfg", r"/model/person/yolo.weights")


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5]
            if scores > CONF_THRESHOLD:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(scores))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, 0.5)
    bboxList = list()

    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        bboxList.append([left, top, left+width, top+height])
        cv2.rectangle(frame, (left, top),
                      (left+width, top+height), (0, 0, 255), 3)

    return frame, bboxList


def detect(img):
    img1 = cv2.resize(img, (320, 320))
    blob = cv2.dnn.blobFromImage(img1, 1/255.0)
    NET.setInput(blob)
    detections = NET.forward(NET.getUnconnectedOutLayersNames())
    img, bboxList = postprocess(img, detections)

    return img, bboxList
