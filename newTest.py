# import cv2
# import time
# import face_recognition
# from skimage.metrics import structural_similarity as ssim
# from PIL import Image
# import imagehash
# import numpy as np
# from keras.models import load_model  # You may need to install keras (pip install keras)

# # Load pre-trained emotion detection model
# emotion_model = load_model('emotion_model.hdf5')

# def faceBox(faceNet, frame):
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

# # motion_model = load_model('emotion_model.h5')  # Replace with the path to your emotion detection model

# def detect_gender_age_emotion(face):
#     # Use OpenCV for gender and age detection
#     gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')  # Replace with your gender detection model files
#     age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')  # Replace with your age detection model files

#     # Gender detection
#     blob_gender = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#     gender_net.setInput(blob_gender)
#     gender_preds = gender_net.forward()
#     gender = "Male" if gender_preds[0][0] > 0.5 else "Female"

#     # Age detection
#     blob_age = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#     age_net.setInput(blob_age)
#     age_preds = age_net.forward()
#     age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'][age_preds[0].argmax()]

#     # Emotion detection
#     gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#     resized_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)
#     resized_face = np.expand_dims(np.expand_dims(resized_face, -1), 0)
#     emotion_preds = emotion_model.predict(resized_face)
#     emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#     emotion = emotion_labels[emotion_preds.argmax()]

#     return gender, age, emotion

# def save_person(image, bbox, count, saved_hashes, saved_faces):
#     x1, y1, x2, y2 = bbox
#     person_region = image[y1:y2, x1:x2]

#     if person_region.size > 0:
#         # Calculate the perceptual hash of the person's face
#         person_hash = imagehash.dhash(Image.fromarray(person_region))
#         is_duplicate = any(person_hash - saved_hash < 5 for saved_hash in saved_hashes)

#         if not is_duplicate:
#             # Detect gender, age, and emotion
#             gender, age, emotion = detect_gender_age_emotion(person_region)

#             # Save the face with a unique filename based on count and timestamp
#             timestamp = int(time.time())
#             filename = f'person_{count}_{timestamp}.jpg'
#             cv2.imwrite(filename, person_region)
#             saved_faces.append({
#                 'filename': filename,
#                 'gender': gender,
#                 'age': age,
#                 'emotion': emotion
#             })
#             saved_hashes.append(person_hash)

# # Initialize face detection model
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"
# faceNet = cv2.dnn.readNet(faceModel, faceProto)

# # Initialize object detection model
# config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model = 'frozen_inference_graph.pb'

# model2=cv2.dnn.readNetFromTensorflow(frozen_model, config_file)
# model = cv2.dnn_DetectionModel(frozen_model, config_file)
# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# num_cameras = 2  # Adjust this based on your setup

# # Initialize cameras
# video_captures = []
# for camera_index in range(num_cameras):
#     video = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
#     video.set(3, 1080)
#     video.set(4, 720)
#     video_captures.append(video)

# # Initialize saved data
# saved_hashes = []
# saved_faces = []

# while True:
#     person_counts = [0] * num_cameras
#     for camera_index, video in enumerate(video_captures):
#         start_time = time.time()
#         ret, img = video.read()

#         # Perform object detection using the model
#         blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
#         model2.setInput(blob)
#         detections = model2.forward()

#         for detection in detections[0, 0, :, :]:
#             class_id = int(detection[1])
#             confidence = detection[2]

#             if class_id == 1 and confidence > 0.5:  # Class 1 corresponds to "person"
#                 person_counts[camera_index] += 1
#                 save_person(img, [int(detection[3] * img.shape[1]), int(detection[4] * img.shape[0]), int(detection[5] * img.shape[1]), int(detection[6] * img.shape[0])], person_counts[camera_index], saved_hashes, saved_faces)

#         cv2.imshow(f"Camera {camera_index}", img)

#         end_time = time.time()
#         detection_time = end_time - start_time
#         print(f"Camera {camera_index} - Detection Time: {detection_time} seconds")
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()


# import cv2
# import time
# import face_recognition
# import tensorflow as tf
# import imagehash
# from PIL import Image
# # from tensorflow.keras.models import load_model
# # from tensorflow.keras.layers import Input
# # from tensorflow.keras.layers.experimental import preprocessing
# import numpy as np
# import os



# def faceBox(faceNet, frame):
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

# def detect_gender_age_emotion(face):
#     # Emotion detection
#     input_tensor = tf.keras.layers.Input(shape=(48, 48, 1))
#     resize_layer = tf.keras.layers.experimental.preprocessing.Resizing(64, 64)(input_tensor)
#     # emotion_model_path = 'emotion_model.h5'  # Replace with the path to your emotion detection model
#     emotion_model = tf.keras.models.load_model("emotion_model.hdf5")
#     emotion_model = tf.keras.models.Model(inputs=input_tensor, outputs=emotion_model(resize_layer))
    
#     face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#     face_resized = cv2.resize(face_gray, (48, 48))
#     face_resized = np.expand_dims(face_resized, axis=-1)  # Add a channel dimension
#     emotion_prediction = emotion_model.predict(np.expand_dims(face_resized, axis=0))
#     predicted_emotion = np.argmax(emotion_prediction)
    
#     emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
#     detected_emotion = emotion_labels[predicted_emotion]
    
#     # Use OpenCV for gender and age detection
#     gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')  # Replace with your gender detection model files
#     age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')  # Replace with your age detection model files

#     # Gender detection
#     blob_gender = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#     gender_net.setInput(blob_gender)
#     gender_preds = gender_net.forward()
#     gender = "Male" if gender_preds[0][0] > 0.5 else "Female"

#     # Age detection
#     blob_age = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
#     age_net.setInput(blob_age)
#     age_preds = age_net.forward()
#     age = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'][age_preds[0].argmax()]

    
#     return gender, age, detected_emotion

# # Paths to models and other configurations
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"
# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"
# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"

# # Load face detection model and other models
# faceNet = cv2.dnn.readNet(faceModel, faceProto)
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)

# # Initialize video capture devices
# num_cameras = 2  # Adjust this based on your setup
# video_captures = [cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) for camera_index in range(num_cameras)]
# for video in video_captures:
#     video.set(3, 1080)
#     video.set(4, 720)

# # Initialize variables to track unique person IDs and saved face images
# person_ids = {}
# saved_images = set()
# saved_hashes = set()

# # Main loop
# while True:
#     person_counts = [0] * num_cameras
#     for camera_index, video in enumerate(video_captures):
#         start_time = time.time()
#         ret, img = video.read()
        
#         # Perform face detection
#         frame, bboxs = faceBox(faceNet, img)
#         recog = face_recognition.face_locations(img)
        
#         # Process each detected face
#         for i, (top, right, bottom, left) in enumerate(recog):
#             # Crop the detected face
#             face = img[top:bottom, left:right]
            
#             # Check if the face belongs to a unique person based on image similarity
#             hash_face = imagehash.dhash(Image.fromarray(face))
#             is_duplicate = any(hash_face - saved_hash < 5 for saved_hash in saved_hashes)
            
#             if is_duplicate:
#                 # This face likely belongs to a person already saved
#                 continue
            
#             # Detect gender, age, and emotion
#             gender, age, emotion = detect_gender_age_emotion(face)
            
#             # Generate a unique person ID
#             if camera_index in person_ids:
#                 person_id = person_ids[camera_index]
#             else:
#                 person_id = len(person_ids) + 1
#                 person_ids[camera_index] = person_id
            
#             # Save the face image with a unique filename
#             if not os.path.exists(f'person_{person_id}'):
#                 os.makedirs(f'person_{person_id}')
#             image_filename = f'person_{person_id}/{time.time()}.jpg'
#             cv2.imwrite(image_filename, face)
            
#             # Record the image hash to avoid duplicates
#             saved_hashes.add(hash_face)
            
#             # Print and display information about the detected person
#             print(f"Camera {camera_index} - Person ID: {person_id}")
#             print(f"Camera {camera_index} - Gender: {gender}, Age: {age}, Emotion: {emotion}")
            
#             # Draw bounding box and labels on the frame
#             cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
#             label = f"Person {person_id}: {gender}, {age}, {emotion}"
#             cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Display the frame
#         cv2.imshow(f"Camera {camera_index}", img)
        
#         end_time = time.time()
#         detection_time = end_time - start_time
#         print(f"Camera {camera_index} - Detection Time: {detection_time} seconds")
    
#     # Check for user input to exit the loop
#     k = cv2.waitKey(1)
#     if k == ord('q'):
#         break

# # Release video captures and close OpenCV windows
# for video in video_captures:
#     video.release()
# cv2.destroyAllWindows()

import cv2
import time
import face_recognition
import numpy as np
import os

# Function for face detection using OpenCV's DNN module
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

# Initialize face detection model
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Initialize video capture devices (adjust camera indices)
num_cameras = 2
video_captures = [cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) for camera_index in range(num_cameras)]
for video in video_captures:
    video.set(3, 1080)
    video.set(4, 720)

# Initialize face recognition
known_face_encodings = []
known_face_names = []

# Add known faces and their names to the lists
# For example:
# known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
# known_face_names.append("John Doe")

# Dictionary to track people across cameras
tracked_people = {}

while True:
    person_counts = [0] * num_cameras
    for camera_index, video in enumerate(video_captures):
        start_time = time.time()
        ret, img = video.read()
        
        # Perform face detection
        frame, bboxs = faceBox(faceNet, img)
        recog = face_recognition.face_locations(img)
        
        # Process each detected face
        for i, (top, right, bottom, left) in enumerate(recog):
            # Crop the detected face
            face = img[top:bottom, left:right]
            
            # Check if the face belongs to a known person
            face_encodings = face_recognition.face_encodings(face)
            if len(face_encodings) == 0:
                continue
            
            match = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
            if any(match):
                # The detected face belongs to a known person
                person_name = known_face_names[match.index(True)]
                person_id = tracked_people.get(person_name)
            else:
                # The detected face is unknown
                person_name = "Unknown"
                person_id = len(tracked_people) + 1
                tracked_people[person_name] = person_id
            
            # Save the face image with a unique filename
            if not os.path.exists(f'person_{person_id}'):
                os.makedirs(f'person_{person_id}')
            image_filename = f'person_{person_id}/{time.time()}.jpg'
            cv2.imwrite(image_filename, face)
            
            # Print and display information about the detected person
            print(f"Camera {camera_index} - Person ID: {person_id}, Name: {person_name}")
            
            # Draw bounding box and labels on the frame
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            label = f"Person {person_id}: {person_name}"
            cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow(f"Camera {camera_index}", img)
        
        end_time = time.time()
        detection_time = end_time - start_time
        print(f"Camera {camera_index} - Detection Time: {detection_time} seconds")
    
    # Check for user input to exit the loop
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release video captures and close OpenCV windows
for video in video_captures:
    video.release()
cv2.destroyAllWindows()
