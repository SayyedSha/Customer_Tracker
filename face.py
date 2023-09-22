# import cv2
# import math
# from deepface import DeepFace
# import numpy as np
# # Load YOLO object detection model
# yolo_model = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# # Load YOLO object detection classes
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")

# # Load gender classification model
# gender_model = DeepFace.build_model("Gender")

# # Load emotion classification model
# emotion_model = DeepFace.build_model("Emotion")

# # Load age classification model
# age_model = DeepFace.build_model("Age")

# # Open a video capture stream (adjust the index as needed)
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(3, 640)
# cap.set(4, 480)

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Perform object detection using YOLO
#     blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
#     yolo_model.setInput(blob)
#     layer_names = yolo_model.getUnconnectedOutLayersNames()
#     detections = yolo_model.forward(layer_names)

#     # Initialize lists to store detected objects, faces, and their properties
#     detected_objects = []
#     detected_faces = []

#     for detection in detections:
#         for obj in detection:
#             scores = obj[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:
#                 if classes[class_id] == "person":
#                     detected_objects.append(obj)

#     # Extract faces from detected objects
#     for obj in detected_objects:
#         x, y, w, h = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
#         face_roi = frame[y:y + h, x:x + w]
#         if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
#             detected_faces.append(face_roi)

#     # Perform face analysis on detected faces
#     for face in detected_faces:
#         # Perform emotion analysis
#         emotion_result = DeepFace.analyze(face, actions=["emotion"])
#         emotion = emotion_result["dominant_emotion"]

#         # Perform age and gender analysis
#         age_gender_result = DeepFace.analyze(face, actions=["age", "gender"])
#         age = age_gender_result["age"]
#         gender = age_gender_result["gender"]

#         # Draw bounding box and labels on the frame
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         label = f"Emotion: {emotion}, Age: {age}, Gender: {gender}"
#         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow("Object Detection & Face Analysis", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video capture and close windows
# cap.release()
# cv2.destroyAllWindows()


import cv2
from deepface import DeepFace
from ultralytics import YOLO
import math


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

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"



faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Open a video capture stream (adjust the index as needed)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1080)
cap.set(4, 720)


model = YOLO("yolov8-Weights/yolov8n.pt")
gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]



while True:
    ret, frame = cap.read()
    results = model(frame, stream=True)

    if not ret:
        break

    # Perform face detection using Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            # face_region = frame[y1:y2, x1:x2]
            # for (x, y, w, h) in faces:
            #     face_roi = frame[y:y + h, x:x + w]
            #     if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
            #         # Perform face analysis (emotion, age, gender)
            #         face_data = DeepFace.analyze(face_region, actions=["emotion", "age", "gender"], enforce_detection=False)
                    
            #         # Extract analysis results
            #         result=face_data[0]
            #         emotion = result["dominant_emotion"]
            #         age = result["age"]
            #         gender = result["dominant_gender"]
                
            #         # Draw bounding box and labels on the frame
            #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #         label = f"Emotion: {emotion}, Age: {age}, Gender: {gender}"
            #         cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2


           
            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(frame,classNames[cls], org, font, fontScale, color, thickness)
    # Display the frame
    cv2.imshow("Face Analysis", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
