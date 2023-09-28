from threading import Thread
import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import face_recognition
import asyncio
import torch


# Load YOLO model
model = YOLO("yolov8n.pt").cuda(device=0)


# Load a pre-trained shape predictor model for face landmark detection
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load a pre-trained face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Directory to save the recognized faces
output_directory = "recognized_faces"
os.makedirs(output_directory, exist_ok=True)



known_face_encoding = []

async def img_dir():
    folder = Path("recognized_faces").glob("*.jpg")
    for images in folder:
        image = face_recognition.load_image_file(images)
        try:
            image_encoded = face_recognition.face_encodings(image)[0]
            known_face_encoding.append(image_encoded)
        except:
            pass

load_known_face_encodings_lambda = lambda: [
                face_recognition.face_encodings(face_recognition.load_image_file(image))[0]
                for image in Path("recognized_faces").glob("*.jpg")
                if face_recognition.face_encodings(face_recognition.load_image_file(image))
            ]

async def face_identifier(frame, counter):
    face = face_recognition.face_locations(frame)
    face_encoded = face_recognition.face_encodings(frame, face)

    for i, faces in enumerate(face_encoded):
        matches = face_recognition.compare_faces(known_face_encoding, faces)

        if True in matches:
            print("Same")
        else:
            for top, right, bottom, left in face:
                person = frame[top:bottom, left:right]
                counter += 1
                unique_filename = f"unknown_face_{counter}_{i}.jpg"
                output_path = os.path.join(output_directory, unique_filename)
                cv2.imwrite(output_path, person)

async def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

faceNet = cv2.dnn.readNet(faceModel, faceProto)

async def face_identifier_2():
    while True:
        try:

            Myframe1 = cam1.getFrame()
            Myframe2 = cam2.getFrame()
            for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
                await face_identifier(frame, counter)

        except:
           print("not working")

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

async def main():
    while True:
        try:
             
            # known_face_encoding=load_known_face_encodings_lambda()

            Myframe1 = cam1.getFrame()
            Myframe2 = cam2.getFrame()

            for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
                await face_identifier_2()
                frames, bboxs = await faceBox(faceNet, frame)

                # await face_identifier(frame, counter)

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

# Run the main asyncio event loop
if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(img_dir())
   





# from threading import Thread
# import cv2
# import dlib
# import numpy as np
# import os
# from ultralytics import YOLO
# from pathlib import Path
# import asyncio
# import torch

# # Load YOLO model
# model = YOLO("yolov8n.pt")


# face_detector = dlib.get_frontal_face_detector()

# # Load a pre-trained shape predictor model for face landmark detection
# shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Load a pre-trained face recognition model from dlib
# face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# # Directory to save the recognized faces
# output_directory = "recognized_faces"
# os.makedirs(output_directory, exist_ok=True)

# known_face_encodings = []

# # Function to load known face encodings
# def load_known_faces():
#     folder_dir = Path("recognized_faces").glob("*.jpg")
#     for image_path in folder_dir:
#         image = cv2.imread(str(image_path))
#         face_encoding = compute_face_encoding(image)
#         if face_encoding is not None:
#             known_face_encodings.append(face_encoding)

# # Function to compute face encodings using dlib
# def compute_face_encoding(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_rectangles = face_detector(gray)
#     if len(face_rectangles) == 0:
#         return None  # No face found
#     landmarks = shape_predictor(gray, face_rectangles[0])
#     face_encoding = np.array(face_recognition_model.compute_face_descriptor(image, landmarks))
#     return face_encoding

# async def face_identifier(frame, counter):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_rectangles = face_detector(gray)
#     for i, rect in enumerate(face_rectangles):
#         x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
#         face_roi = frame[y1:y2, x1:x2]
#         face_encoding = compute_face_encoding(face_roi)
#         if face_encoding is not None:
#             for known_encoding in known_face_encodings:
#                 distance = np.linalg.norm(face_encoding - known_encoding)
#                 if distance < 0.6:  # Adjust the threshold as needed
#                     print("Recognized")
#                     break
#             else:
#                 counter += 1
                
#                 unique_filename = f"unknown_face_{counter}_{i}.jpg"
#                 output_path = os.path.join(output_directory, unique_filename)
#                 cv2.imwrite(output_path, face_roi)

# # Rest of your code remains the same
# async def faceBox(faceNet, frame):
#     frameHeight = frame.shape[0]
#     frameWidth = frame.shape[1]
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
#     faceNet.setInput(blob)
#     detection = faceNet.forward()
#     bboxs = []
#     for i in range(detection.shape[2]):
#         confidence = detection[0, 0, i, 2]
#         if confidence > 0.7:
#             x1 = int(detection[0, 0, i, 3] * frameWidth)
#             y1 = int(detection[0, 0, i, 4] * frameHeight)
#             x2 = int(detection[0, 0, i, 5] * frameWidth)
#             y2 = int(detection[0, 0, i, 6] * frameHeight)
#             bboxs.append([x1, y1, x2, y2])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     return frame, bboxs

# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# faceNet = cv2.dnn.readNet(faceModel, faceProto)

# class VStream:
#     def __init__(self, src, unique_id=None):
#         self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()
#         self.unique_id = unique_id

#     def update(self):
#         while True:
#             _, self.frame = self.capture.read()

#     def getFrame(self):
#         return self.frame

# flip = 2
# dispW = 640
# dispH = 480

# # Create a dictionary to store recognized faces and their unique IDs
# recognized_faces = {}
# next_object_id = 1
# padding = 20

# # Initialize a variable to store the unique ID for recognized faces from Camera 0
# unique_id_camera_0 = None

# cam1 = VStream(0, unique_id=unique_id_camera_0)  # Camera 0 with unique ID
# cam2 = VStream(1)  # Camera 1 with no unique ID initially

# frame_skip_counter = 0
# global counter
# counter=0

# # Main asyncio event loop
# async def main():
#     load_known_faces()
#     while True:
#         try:
#             Myframe1 = cam1.getFrame()
#             Myframe2 = cam2.getFrame()

#             for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
#                 frames, bboxs = await faceBox(faceNet, frame)
#                 await face_identifier(frame, counter)

#                 for bbox in bboxs:
#                     face = frames[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#                     face = frames[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
#                             max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
#                     cv2.rectangle(frames, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)

#                 results = model.track(frame, persist=True)
#                 annotated_frame = results[0].plot()

#                 cv2.imshow(frame_name, annotated_frame)
#         except Exception as e:
#             print(f'Error: {str(e)}')

#         if cv2.waitKey(1) == ord('q'):
#             cam1.capture.release()
#             cam2.capture.release()
#             cv2.destroyAllWindows()
#             break

# if __name__ == "__main__":
#     asyncio.run(main())

# from threading import Thread
# import cv2
# import dlib
# import numpy as np
# import os
# from pathlib import Path
# import asyncio

# # Load a pre-trained deep learning face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load a pre-trained shape predictor model for face landmark detection
# shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Load a pre-trained face recognition model from dlib
# face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# # Directory to save the recognized faces
# output_directory = "recognized_faces"
# os.makedirs(output_directory, exist_ok=True)

# known_face_encodings = []

# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# faceNet = cv2.dnn.readNet(faceModel, faceProto)


# # Function to load known face encodings
# def load_known_faces():
#     folder_dir = Path("recognized_faces").glob("*.jpg")
#     for image_path in folder_dir:
#         image = cv2.imread(str(image_path))
#         face_encoding = compute_face_encoding(image)
#         if face_encoding is not None:
#             known_face_encodings.append(face_encoding)

# # Function to compute face encodings using dlib
# def compute_face_encoding(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face_rectangles = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     if len(face_rectangles) == 0:
#         return None  # No face found
#     x, y, w, h = face_rectangles[0]
#     face_roi = gray[y:y + h, x:x + w]
#     landmarks = shape_predictor(face_roi, dlib.rectangle(0, 0, w, h))
#     face_encoding = np.array(face_recognition_model.compute_face_descriptor(image, landmarks))
#     return face_encoding

# async def face_identifier(frame, counter):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_rectangles = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     for i, (x, y, w, h) in enumerate(face_rectangles):
#         face_roi = frame[y:y + h, x:x + w]
#         face_encoding = compute_face_encoding(face_roi)
#         if face_encoding is not None:
#             for known_encoding in known_face_encodings:
#                 distance = np.linalg.norm(face_encoding - known_encoding)
#                 if distance < 0.6:  # Adjust the threshold as needed
#                     print("Recognized")
#                     break
#             else:
#                 counter += 1
#                 unique_filename = f"unknown_face_{counter}_{i}.jpg"
#                 output_path = os.path.join(output_directory, unique_filename)
#                 cv2.imwrite(output_path, face_roi)


# async def faceBox(faceNet, frame):
#     frameHeight = frame.shape[0]
#     frameWidth = frame.shape[1]
    
#     # Convert the grayscale frame to a 3-channel BGR image
#     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
#     faceNet.setInput(blob)
#     detection = faceNet.forward()
#     bboxs = []
#     for i in range(detection.shape[2]):
#         confidence = detection[0, 0, i, 2]
#         if confidence > 0.7:
#             x1 = int(detection[0, 0, i, 3] * frameWidth)
#             y1 = int(detection[0, 0, i, 4] * frameHeight)
#             x2 = int(detection[0, 0, i, 5] * frameWidth)
#             y2 = int(detection[0, 0, i, 6] * frameHeight)
#             bboxs.append([x1, y1, x2, y2])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
#     return frame, bboxs


# class VStream:
#     def __init__(self, src, unique_id=None):
#         self.capture = cv2.VideoCapture(src)
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()
#         self.unique_id = unique_id

#     def update(self):
#         while True:
#             _, self.frame = self.capture.read()

#     def getFrame(self):
#         return self.frame

# dispW = 640
# dispH = 480

# # Create a dictionary to store recognized faces and their unique IDs
# recognized_faces = {}
# next_object_id = 1
# padding = 20

# # Initialize a variable to store the unique ID for recognized faces from Camera 0
# unique_id_camera_0 = None

# cam1 = VStream(0, unique_id=unique_id_camera_0)  # Camera 0 with unique ID
# cam2 = VStream(1)  # Camera 1 with no unique ID initially

# frame_skip_counter = 0
# global counter
# counter = 0

# # Main asyncio event loop
# async def main():
#     load_known_faces()
#     while True:
#         try:
#             Myframe1 = cam1.getFrame()
#             Myframe2 = cam2.getFrame()

#             for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
#                 gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 frames, bboxs = await faceBox(faceNet,gray_frame)
#                 await face_identifier(frame, counter)

#                 for bbox in bboxs:
#                     face = frames[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#                     face = frames[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
#                             max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
#                     cv2.rectangle(frames, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)

#                 cv2.imshow(frame_name, frames)
#         except Exception as e:
#             print(f'Error: {str(e)}')

#         if cv2.waitKey(1) == ord('q'):
#             cam1.capture.release()
#             cam2.capture.release()
#             cv2.destroyAllWindows()
#             break

# if __name__ == "__main__":
#     asyncio.run(main())
