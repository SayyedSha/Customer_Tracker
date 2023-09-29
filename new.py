# from threading import Thread
# import cv2
# import dlib
# import numpy as np
# import os
# from ultralytics import YOLO
# from pathlib import Path
# import face_recognition
# import asyncio
# import torch

# # Load YOLO model
# model = YOLO("yolov8n.pt").cuda(device=0)

# # Load a pre-trained shape predictor model for face landmark detection
# shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Load a pre-trained face detector from dlib
# face_detector = dlib.get_frontal_face_detector()

# # Directory to save the recognized faces
# output_directory = "recognized_faces"
# os.makedirs(output_directory, exist_ok=True)
# class FaceRecognitionManager:
#     def __init__(self):
#         self.known_faces = {}  # Dictionary to store known faces
#         self.next_id = 1  # Next available unique ID

#     # Function to load known face encodings
#     async def load_known_faces(self):
#         folder = Path("recognized_faces").glob("*.jpg")
#         for image_path in folder:
#             image = face_recognition.load_image_file(image_path)
#             try:
#                 encoding = face_recognition.face_encodings(image)[0]
#                 self.known_faces[image_path.stem] = {
#                     'encoding': encoding,
#                     'id': self.next_id
#                 }
#                 self.next_id += 1
#             except:
#                 pass

#     # Function to identify faces
#     async def identify_faces(self, frame):
#         face_locations = face_recognition.face_locations(frame)
#         face_encodings = face_recognition.face_encodings(frame, face_locations)

#         for i, encoding in enumerate(face_encodings):
#             matches = []

#             for name, known_face in self.known_faces.items():
#                 face_to_compare = known_face['encoding']
#                 match = face_recognition.compare_faces([face_to_compare], encoding)
#                 matches.append((name, match))

#             for name, match in matches:
#                 if True in match:
#                     print(f"Recognized: {name} (ID: {self.known_faces[name]['id']})")
#                     # You can use the recognized ID to track the person
#                 else:
#                     print("Unknown")

# # Create an instance of the FaceRecognitionManager
# face_recognition_manager = FaceRecognitionManager()

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
# counter = 0

# # Main async function
# async def main():
#     # await face_recognition_manager.load_known_faces()  # Load known face encodings

#     while True:
#         try:

#             Myframe1 = cam1.getFrame()
#             Myframe2 = cam2.getFrame()

#             for frame, frame_name in [(Myframe1, 'Camera 1'), (Myframe2, 'Camera 2')]:
#                 frames, bboxs = await faceBox(faceNet, frame)
#                 # await face_recognition_manager.identify_faces(frame)

#                 for bbox in bboxs:
#                     face = frames[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#                     cv2.rectangle(frames, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)

#                 results = model.track(frame, persist=True)
#                 annotated_frame = results[0].plot()

#                 cv2.imshow(frame_name, annotated_frame)
#         except Exception as e:
#             print(f"Error {e}")
            
#         if cv2.waitKey(1) == ord('q'):
#             cam1.capture.release()
#             cam2.capture.release()
#             cv2.destroyAllWindows()
#             break

# # Run the main asyncio event loop
# if __name__ == "__main__":
#     asyncio.run(main())


from threading import Thread
import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import asyncio
import torch
import face_recognition
import time

# Load YOLO model
model = YOLO("yolov8n.pt").cuda(device=0)


face_detector = dlib.get_frontal_face_detector()

# Load a pre-trained shape predictor model for face landmark detection
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load a pre-trained face recognition model from dlib
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Directory to save the recognized faces
output_directory = "recognized_faces"
os.makedirs(output_directory, exist_ok=True)

known_face_encoding = []

def set_encoding(encoding):
    global known_face_encoding
    known_face_encoding.append(encoding)
    
def get_encoding():
    return known_face_encoding


# Function to load known face encodings
async def img_dir():
    folder_dir = Path("recognized_faces").glob("*.jpg")
    for images in folder_dir:
        image = face_recognition.load_image_file(images)
        try:
            image_encoded = face_recognition.face_encodings(image)[0]
            known_face_encoding.append(image_encoded)
        except:
            pass



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


async def face_identifier_2(cam1,cam2):
    while True:
        try:
            await img_dir()
            Myframe1 = cam1.getFrame()
            Myframe2 = cam2.getFrame()
            for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
                await face_identifier(frame, counter)

        except:
           print("not working")

def faceBox(faceNet, frame):
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

class VStream:
    def __init__(self, src, unique_id=None):
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.fps=int(self.capture.get(cv2.CAP_PROP_FPS))
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.unique_id = unique_id

    def update(self):
        while True:
            _, self.frame = self.capture.read()

    def getFrame(self):
        return self.frame
    def getFPS(self):
        return self.fps

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

frame_count = 0

fps2=0

# from face_saver import face_Saver
# Main asyncio event loop
async def main0():
    
    while True:
        
        try:
          
            Myframe1 = cam1.getFrame()
            Myframe2 = cam2.getFrame()
            

            for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
                frames, bboxs = faceBox(faceNet, frame)
                
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



async def main1():
    task1=asyncio.create_task(main0())
    task2=asyncio.create_task(face_identifier_2(cam1,cam2))

    await task1
    await task2
if __name__ == "__main__":
   
    asyncio.run(main1())
    




