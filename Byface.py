# from threading import Thread
# import cv2
# from ultralytics import YOLO
# import face_recognition
# import numpy as np

# model = YOLO("yolov8n.pt")

# class VStream:
#     def __init__(self, src):
#         self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()

#     def update(self):
#         while True:
#             _, self.frame = self.capture.read()

#     def getFrame(self):
#         return self.frame

# flip = 2
# dispW = 640
# dispH = 480

# cam1 = VStream(0)
# cam2 = VStream(1)

# # Create a dictionary to store recognized faces and their unique IDs
# recognized_faces = {}
# next_object_id = 1  

# while True:
#     try:
#         Myframe1 = cam1.getFrame()
#         Myframe2 = cam2.getFrame()

#         for frame, frame_name in [(Myframe1, 'Camera 1'), (Myframe2, 'Camera 2')]:
#             results = model.track(frame, persist=True)
#             annotated_frame = results[0].plot()

#             for item in results[0].names:
#                 if item in recognized_faces:
#                     face_locations = face_recognition.face_locations(frame)
#                     face_encodings = face_recognition.face_encodings(frame, [face_locations[0]])
#                     unique_id, face_encoding = recognized_faces[item]
#                     if face_encodings==face_encoding:
#                         print("Same Person")
#                     else:
#                         print("differnt person")
                
#                 else:
#                     # Assign a new unique ID for faces not recognized yet
#                     unique_id = next_object_id
#                     face_locations = face_recognition.face_locations(frame)
                    
#                     if len(face_locations) > 0:
#                         # Take the first detected face for simplicity
#                         face_encodings = face_recognition.face_encodings(frame, [face_locations[0]])
#                         if len(face_encodings) > 0:
#                             face_encoding = face_encodings[0]
#                             recognized_faces[item] = (unique_id, face_encoding)
#                             next_object_id += 1

#                 cv2.putText(annotated_frame, f"ID: {unique_id}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
#             cv2.imshow(frame_name, annotated_frame)
#     except Exception as e:
#         print(f'Error: {str(e)}')

#     if cv2.waitKey(1) == ord('q'):
#         cam1.capture.release()
#         cam2.capture.release()
#         cv2.destroyAllWindows()
#         break


from threading import Thread
import cv2
from ultralytics import YOLO
import face_recognition
import numpy as np

model = YOLO("yolov8n.pt")

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

flip = 2
dispW = 640
dispH = 480

cam1 = VStream(0)
cam2 = VStream(1)

# Create a dictionary to store recognized faces and their unique IDs
recognized_faces = {}
next_object_id = 1  

while True:
    try:
        Myframe1 = cam1.getFrame()
        Myframe2 = cam2.getFrame()

        for frame, frame_name in [(Myframe1, 'Camera 1'), (Myframe2, 'Camera 2')]:
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            for item in results[0].names:
                if item in recognized_faces:
                    face_locations = face_recognition.face_locations(frame)
                    face_encodings = face_recognition.face_encodings(frame, [face_locations[0]])
                    unique_id, stored_face_encoding = recognized_faces[item]
                    if len(face_encodings) > 0:
                        current_face_encoding = face_encodings[0]
                        # Compare the current face encoding to the stored one
                        is_same_person = face_recognition.compare_faces([stored_face_encoding], current_face_encoding)
                        if is_same_person[0]:
                            print("Same Person")
                        else:
                            print("Different person")
                
                else:
                    # Assign a new unique ID for faces not recognized yet
                    unique_id = next_object_id
                    face_locations = face_recognition.face_locations(frame)
                    
                    if len(face_locations) > 0:
                        # Take the first detected face for simplicity
                        face_encodings = face_recognition.face_encodings(frame, [face_locations[0]])
                        if len(face_encodings) > 0:
                            face_encoding = face_encodings[0]
                            recognized_faces[item] = (unique_id, face_encoding)
                            next_object_id += 1

                cv2.putText(annotated_frame, f"ID: {unique_id}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow(frame_name, annotated_frame)
    except Exception as e:
        print(f'Error: {str(e)}')

    if cv2.waitKey(1) == ord('q'):
        cam1.capture.release()
        cam2.capture.release()
        cv2.destroyAllWindows()
        break
