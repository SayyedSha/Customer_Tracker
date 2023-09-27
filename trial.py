from threading import Thread
import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import face_recognition
import asyncio

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load a pre-trained shape predictor model for face landmark detection
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load a pre-trained face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Directory to save the recognized faces
output_directory = "recognized_faces"
os.makedirs(output_directory, exist_ok=True)

folder_dir=Path("recognized_faces").glob("*.jpg")

known_face_encoding=[]

async def img_dir():
    folder_dir=Path("recognized_faces").glob("*.jpg")
    for images in folder_dir:
        image=face_recognition.load_image_file(images)
        try:
            image_encoded=face_recognition.face_encodings(image)[0]
            known_face_encoding.append(image_encoded)
        except:
            pass

async def face_identifier(frame):
    face=face_recognition.face_locations(frame)
    face_encoded=face_recognition.face_encodings(frame,face)

    for i, faces in enumerate(face_encoded):
        # print(i)
        matches=face_recognition.compare_faces(known_face_encoding,faces)

        if True in matches:
            print ("Same")
        else:
            
            for top, right, bottom, left in face:
                person = frame[top:bottom, left:right]
                counter+=1
                unique_filename = f"unknown_face_{counter}_{i}.jpg"
                output_path = os.path.join(output_directory, unique_filename)
                # recognized_faces.update(unique_id=unique_id)
                cv2.imwrite(output_path, person)

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
counter=0

while True:
    try:
        asyncio.run(img_dir())

        Myframe1 = cam1.getFrame()
        Myframe2 = cam2.getFrame()

        for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
            frames, bboxs = faceBox(faceNet, frame)
    
            asyncio.run(face_identifier(frame))
                

            # if unique_id is not None:
            #     # Perform face recognition with dlib
            #     # known_face=
            #     faces = face_detector(frames)

            #     for face in faces:
            #         landmarks = shape_predictor(frames, face)
            #         landmarks_np = np.array([[p.x, p.y] for p in landmarks.parts()])

            #         # Compare with known faces
            #         found = False
            #         for known_id, known_landmarks in recognized_faces.items():
            #             distance = np.linalg.norm(landmarks_np - known_landmarks)

            #             # If the distance is below a threshold, consider it a match
            #             if distance < 0.6:
            #                 found = True
            #                 unique_id = known_id
            #                 break

            #         # If not found in recognized faces, assign a new ID
            #         if not found:
            #             recognized_faces[next_object_id] = landmarks_np
            #             unique_id = next_object_id
            #             next_object_id += 1

            #         # Store the recognized face and its unique ID
            #         if unique_id is not None:
            #             # Crop the recognized face
            #             x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            #             cropped_face = frames[y1:y2, x1:x2]

            #             # Save the cropped face with the unique ID
            #             output_path = os.path.join(output_directory, f"face_{unique_id}.jpg")
            #             cv2.imwrite(output_path, cropped_face)

                # frame_skip_counter += 1

            if frame_skip_counter % 5 == 0:
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
