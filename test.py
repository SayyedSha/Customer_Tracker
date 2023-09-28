# import time
# from deepface import DeepFace
# from ultralytics import YOLO
# import cv2
# import math
# import face_recognition as fr


# # Create VideoCapture objects for multiple cameras
# cameras = [
#     cv2.VideoCapture(0, cv2.CAP_DSHOW),  # Camera 1
#     cv2.VideoCapture(1, cv2.CAP_DSHOW),  # Camera 2
#     # Add more cameras if needed
# ]

# for cap in cameras:
#     cap.set(3, 640)
#     cap.set(4, 480)

# # Load YOLO model
# model = YOLO("yolo-Weights/yolov8n.pt")
# gender_model = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# # Define object classes
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"]


# def faceBox(faceNet,frame):
#     frameHeight=frame.shape[0]
#     frameWidth=frame.shape[1]
#     blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
#     faceNet.setInput(blob)
#     detection=faceNet.forward()
#     bboxs=[]
#     for i in range(detection.shape[2]):
#         confidence=detection[0,0,i,2]
#         if confidence>0.9:
#             x1=int(detection[0,0,i,3]*frameWidth)
#             y1=int(detection[0,0,i,4]*frameHeight)
#             x2=int(detection[0,0,i,5]*frameWidth)
#             y2=int(detection[0,0,i,6]*frameHeight)
#             bboxs.append([x1,y1,x2,y2])
#             cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
#     return frame, bboxs


# # Face detection model files
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# # Age and gender detection model files
# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"
# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"

# # Load face detection model
# faceNet = cv2.dnn.readNet(faceModel, faceProto)

# # Load age and gender detection models
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

# padding = 20

# while True:
#     for idx, cap in enumerate(cameras):
#         start_time = time.time()
#         success, img = cap.read()
#         results = model(img, stream=True)
#         frame, bboxs = faceBox(faceNet, img)

#         for bbox in bboxs:
#             face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
#                         max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
#             blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#             genderNet.setInput(blob)
#             genderPred = genderNet.forward()
#             gender = genderList[genderPred[0].argmax()]

#             ageNet.setInput(blob)
#             agePred = ageNet.forward()
#             age = ageList[agePred[0].argmax()]

#             label = "{},{}".format(gender, age)
#             cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
#             cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
#                         cv2.LINE_AA)

#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 face_region = img[y1:y2, x1:x2]
#                 confidence = math.ceil((box.conf[0] * 100)) / 100
#                 cls = int(box.cls[0])
#                 org = [x1, y1]
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 fontScale = 1
#                 color = (255, 0, 0)
#                 thickness = 2
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
#                 cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

#         end_time = time.time()
#         detection_time = end_time - start_time
#         print(f"Camera {idx + 1} Detection Time: {detection_time:.2f} seconds")

#         cv2.imshow(f'Webcam {idx + 1}', img)

#     if cv2.waitKey(1) == ord('q'):
#         break

# for cap in cameras:
#     cap.release()
# cv2.destroyAllWindows()


from threading import Thread
import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import asyncio
import torch
from multiprocessing import Process, Manager

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load a pre-trained shape predictor model for face landmark detection
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load a pre-trained face recognition model from dlib
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory to save the recognized faces
output_directory = "recognized_faces"
os.makedirs(output_directory, exist_ok=True)

known_face_encodings = []

# Function to load known face encodings
def load_known_faces():
    folder_dir = Path("recognized_faces").glob("*.jpg")
    for image_path in folder_dir:
        image = cv2.imread(str(image_path))
        face_encoding = compute_face_encoding(image)
        if face_encoding is not None:
            known_face_encodings.append(face_encoding)

# Function to compute face encodings using dlib
def compute_face_encoding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rectangles = face_detector(gray)
    if len(face_rectangles) == 0:
        return None  # No face found
    landmarks = shape_predictor(gray, face_rectangles[0])
    face_encoding = np.array(face_recognition_model.compute_face_descriptor(image, landmarks))
    return face_encoding

# Function for face recognition and comparison
def face_recognition_worker(frame, counter, known_face_encodings, result_dict):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rectangles = face_detector(gray)
    for i, rect in enumerate(face_rectangles):
        face_encoding = compute_face_encoding(frame)
        if face_encoding is not None:
            for known_encoding in known_face_encodings:
                distance = np.linalg.norm(face_encoding - known_encoding)
                if distance < 0.6:  # Adjust the threshold as needed
                    result_dict[counter] = "Recognized"
                    break
            else:
                result_dict[counter] = "Unknown"
                counter += 1
                unique_filename = f"unknown_face_{counter}_{i}.jpg"
                output_path = os.path.join(output_directory, unique_filename)
                cv2.imwrite(output_path, frame)

# Rest of your code remains the same
async def faceBox(faceNet, frame):
    # Implement your face detection logic here

# Define a class for camera streaming
class VStream:
    def __init__(self, src, unique_id=None):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.unique_id = unique_id

    def update(self):
        while True:
            _, self.frame = self.capture.read()

    def getFrame(self):
        return self.frame

flip = 2
dispW = 640
dispH = 480

# Create a dictionary to store recognized faces and their unique IDs
recognized_faces = {}
next_object_id = 1
padding = 20

# Initialize a variable to store the unique ID for recognized faces from Camera 0
unique_id_camera_0 = None

cam1 = VStream(0, unique_id=unique_id_camera_0)  # Camera 0 with unique ID
cam2 = VStream(1)  # Camera 1 with no unique ID initially

frame_skip_counter = 0
counter = 0

# Main asyncio event loop
async def main():
    load_known_faces()
    while True:
        try:
            Myframe1 = cam1.getFrame()
            Myframe2 = cam2.getFrame()

            for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
                counter += 1
                process = Process(target=face_recognition_worker, args=(frame, counter, known_face_encodings, result_dict))
                process.start()
                process.join()

                frames, bboxs = await faceBox(faceNet, frame)

                for bbox in bboxs:
                    face = frames[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    face = frames[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
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

if __name__ == "__main__":
    asyncio.run(main())

from threading import Thread
import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import asyncio
import torch
from multiprocessing import Process, Manager

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load a pre-trained shape predictor model for face landmark detection
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load a pre-trained face recognition model from dlib
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory to save the recognized faces
output_directory = "recognized_faces"
os.makedirs(output_directory, exist_ok=True)

known_face_encodings = []

# Function to load known face encodings
def load_known_faces():
    folder_dir = Path("recognized_faces").glob("*.jpg")
    for image_path in folder_dir:
        image = cv2.imread(str(image_path))
        face_encoding = compute_face_encoding(image)
        if face_encoding is not None:
            known_face_encodings.append(face_encoding)

# Function to compute face encodings using dlib
def compute_face_encoding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rectangles = face_detector(gray)
    if len(face_rectangles) == 0:
        return None  # No face found
    landmarks = shape_predictor(gray, face_rectangles[0])
    face_encoding = np.array(face_recognition_model.compute_face_descriptor(image, landmarks))
    return face_encoding

# Function for face recognition and comparison
def face_recognition_worker(frame, result_dict, counter):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rectangles = face_detector(gray)
    for i, rect in enumerate(face_rectangles):
        face_encoding = compute_face_encoding(frame)
        if face_encoding is not None:
            for known_encoding in known_face_encodings:
                distance = np.linalg.norm(face_encoding - known_encoding)
                if distance < 0.6:  # Adjust the threshold as needed
                    result_dict[counter] = "Recognized"
                    break
            else:
                with counter.get_lock():
                    counter.value += 1
                unique_filename = f"unknown_face_{counter.value}_{i}.jpg"
                output_path = os.path.join(output_directory, unique_filename)
                cv2.imwrite(output_path, frame)
                result_dict[counter.value] = "Unknown"

# Rest of your code remains the same
async def faceBox(faceNet, frame):
    # Implement your face detection logic here

# Define a class for camera streaming
class VStream:
    def __init__(self, src, unique_id=None):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.unique_id = unique_id

    def update(self):
        while True:
            _, self.frame = self.capture.read()

    def getFrame(self):
        return self.frame

flip = 2
dispW = 640
dispH = 480

# Create a dictionary to store recognized faces and their unique IDs
recognized_faces = {}
next_object_id = 1
padding = 20

# Initialize a variable to store the unique ID for recognized faces from Camera 0
unique_id_camera_0 = None

cam1 = VStream(0, unique_id=unique_id_camera_0)  # Camera 0 with unique ID
cam2 = VStream(1)  # Camera 1 with no unique ID initially

frame_skip_counter = 0

# Use a multiprocessing manager to create a shared counter
manager = Manager()
counter = manager.Value('i', 0)

# Use a multiprocessing manager to create a shared dictionary for results
result_dict = manager.dict()

# Main asyncio event loop
async def main():
    load_known_faces()
    while True:
        try:
            Myframe1 = cam1.getFrame()
            Myframe2 = cam2.getFrame()

            for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
                counter.value += 1
                process = Process(target=face_recognition_worker, args=(frame, result_dict, counter))
                process.start()
                process.join()

                frames, bboxs = await faceBox(faceNet, frame)

                for bbox in bboxs:
                    face = frames[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    face = frames[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
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

if __name__ == "__main__":
    asyncio.run(main())
