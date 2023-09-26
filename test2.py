# import face_recognition
# import cv2

# camera1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Camera 1
# camera2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # Camera 2

# # Load a sample image with known faces (for face recognition)
# # known_image = face_recognition.load_image_file("known_person.jpg")
# # known_face_encoding = face_recognition.face_encodings(known_image)[0]

# while True:
#     ret1, frame1 = camera1.read()
#     ret2, frame2 = camera2.read()

#     if not ret1 or not ret2:
#         print("Error reading frames from one or both cameras")
#         break

#     # Detect faces in each frame
#     face_locations1 = face_recognition.face_locations(frame1)
#     face_locations2 = face_recognition.face_locations(frame2)

#     # You can now draw rectangles around the detected faces
#     for top, right, bottom, left in face_locations1:
#         cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

#     for top, right, bottom, left in face_locations2:
#         cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 255), 2)

#     cv2.imshow("Camera 1", frame1)
#     cv2.imshow("Camera 2", frame2)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# camera1.release()
# camera2.release()
# cv2.destroyAllWindows()




from threading import Thread
import cv2
import dlib
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import face_recognition


# Load YOLO model
model = YOLO("yolov8n.pt")

# Load a pre-trained shape predictor model for face landmark detection
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load a pre-trained face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Directory to save the recognized faces
output_folder = "recognized_faces"
os.makedirs(output_folder, exist_ok=True)

folder_dir=Path("recognized_faces").glob("*jpg")

# image_files = [os.path.join(folder_dir, file) for file in os.listdir(folder_dir) if file.endswith(('.jpg', '.png', '.jpeg'))]

known_face_encoding=[]

known_face_images = []
known_face_names = []

def load_known_faces(folder_dir):
    for image_path in folder_dir:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        known_face_names.append(image_name)
        known_face_images.append(cv2.imread(str(image_path)))



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
reference_embeddings = {}
counter=0
while True:
    try:
        folder_dir=Path("recognized_faces").glob("*jpg")
        load_known_faces(folder_dir)
        print(known_face_images)

        Myframe1 = cam1.getFrame()
        Myframe2 = cam2.getFrame()

        for frame, frame_name, unique_id in [(Myframe1, 'Camera 1', unique_id_camera_0), (Myframe2, 'Camera 2', None)]:
            frames, bboxs = faceBox(faceNet, frame) 
    
            if frame_skip_counter % 5 == 0:
                frames, bboxs = faceBox(faceNet, frame)
                

                for bbox in bboxs:
                    face = frames[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    face = frames[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
                    cv2.rectangle(frames, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
                    
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                match_found = False
                for known_face_image, known_face_name in zip(known_face_images, known_face_names):
                    # Convert the known face to grayscale
                    known_face_gray = cv2.cvtColor(known_face_image, cv2.COLOR_BGR2GRAY)

                    # Use a face recognition method (e.g., LBPH) for comparison
                    # You can replace this with your preferred face recognition method
                    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                    face_recognizer.train([known_face_gray], np.array([0]))
                    label, confidence = face_recognizer.predict(face_gray)

                    # Set a confidence threshold for face recognition
                    confidence_threshold = 60  # Adjust this threshold as needed

                    if confidence < confidence_threshold:
                        # If a match is found, update the recognized_faces dictionary
                        match_found = True
                        recognized_faces[label] = known_face_name

                # If no match was found, save the unknown face to the output directory
                if not match_found:
                    unique_filename = f"unknown_face_{next_object_id}.jpg"
                    next_object_id += 1
                    output_path = os.path.join(output_folder, unique_filename)
                    cv2.imwrite(output_path, face)
                    
            # results = model.track(frame, persist=True)
            # annotated_frame = results[0].plot()

            cv2.imshow(frame_name)
    except Exception as e:
        print(f'Error: {str(e)}')

    if cv2.waitKey(1) == ord('q'):
        cam1.capture.release()
        cam2.capture.release()
        cv2.destroyAllWindows()
        break
