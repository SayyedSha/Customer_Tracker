from threading import Thread
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

faceNet=cv2.dnn.readNet(faceModel, faceProto)



class VStream:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            _, self.frame = self.capture.read()

    def getFrame(self):
        return self.frame

dispW = 640
dispH = 480

cam1 = VStream(0)
cam2 = VStream(1)

padding = 20
frame_skip_counter = 0

while True:
    try:
        Myframe1 = cam1.getFrame()
        Myframe2 = cam2.getFrame()

        for frame, frame_name in [(Myframe1, 'Camera 1'), (Myframe2, 'Camera 2')]:
            frame_skip_counter += 1

            if frame_skip_counter % 5 == 0:
                frames, bboxs = faceBox(faceNet,frame)

                for bbox in bboxs:
                    cv2.rectangle(frames, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)

            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            cv2.imshow(frame_name, annotated_frame)
    except Exception as e:
        print(f'Error: {str(e)}')

    if cv2.waitKey(1) == ord('q'):
        cam1.capture.release()
        cam2.capture.release()
        cv2.destroyAllWindows()
        break
