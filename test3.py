
#Main Code

# from threading import Thread
# import cv2
# import time
# from ultralytics import YOLO
# import face_recognition


# # config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# # frozen_model='frozen_inference_graph.pb'
# # model=cv2.dnn_DetectionModel(frozen_model,config_file)
# # classlabel=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

# # model.setInputSize(320, 320)
# # model.setInputScale(1.0 / 127.5)
# # model.setInputMean((127.5, 127.5, 127.5))
# # model.setInputSwapRB(True)

# # unique_ids = {}

# # def assign_unique_id(cIndex, frame_name):
# #     global unique_ids
# #     if frame_name == 'Camera 1':
# #         if cIndex not in unique_ids:
# #             # Generate a new unique ID for the person in Camera 1
# #             unique_ids[cIndex] = len(unique_ids) + 1
# #         return unique_ids[cIndex]
# #     elif frame_name == 'Camera 2':
# #         if cIndex in unique_ids:
# #             # Use the same unique ID assigned in Camera 1 for the person in Camera 2
# #             return unique_ids[cIndex]
# #         else:
# #             # Generate a new unique ID for the person in Camera 2
# #             new_id = max(unique_ids.values()) + 1 if unique_ids else 1
# #             unique_ids[cIndex] = new_id
# #             return new_id

# model=YOLO("yolov8n.pt")
# class VStream:
#     def __init__(self,src):
#         self.capture=cv2.VideoCapture(src,cv2.CAP_DSHOW)
#         self.thread=Thread(target=self.update,args=())
#         self.thread.daemon=True
#         self.thread.start()
#         # self.persons = {} 
#     def update(self):
#         while True:
#             _,self.frame=self.capture.read()
#     def getFrame(self):
#         return self.frame
    
# flip=2
# dispW=640
# dispH=480

# cam1=VStream(0)
# cam2=VStream(1)

# object_ids_camera0 = {}  # Dictionary to store object IDs for Camera 0
# next_object_id = 1  

# while True:
#     try:
#         Myframe1=cam1.getFrame()
#         Myframe2=cam2.getFrame()

#         for frame, frame_name in [(Myframe1, 'Camera 1'), (Myframe2, 'Camera 2')]:
#                     results=model.track(frame,persist=True)
#                     print(results)
#                     annotated_frame = results[0].plot()




#                     for item in results[0].names:
#                         if item in object_ids_camera0:
                            
#                             unique_id = object_ids_camera0[item]
#                         else:
#                             # Assign a new unique ID for objects in Camera 0
#                             unique_id = next_object_id
#                             # face=face_recognition.face_locations(frame)

#                             object_ids_camera0[item] = unique_id
#                             next_object_id += 1

#                         cv2.putText(annotated_frame, f"ID: {unique_id}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
#                     # classIndex, Confidence, box = model.detect(frame, confThreshold=0.5)
#                     # if len(classIndex) != 0:
#                     #     for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(), box):
#                     #         # if cIndex <= 9:
#                     #         if classlabel[cIndex - 1] == "person":
#                     #             unique_id = assign_unique_id( cIndex, frame_name)  # Assign or retrieve unique ID
#                     #             cv2.rectangle(frame, boxes, (255, 0, 0), 2)
#                     #             cv2.putText(frame, f"Person {unique_id}", (boxes[0] + 10, boxes[1] + 40),
#                     #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)
                    
#                     # cv2.imshow(frame_name, frame)
#                     cv2.imshow(frame_name, annotated_frame)
#     except:
#         print('Frame not Available')

#     if cv2.waitKey(1)==ord('q'):
#         cam1.capture.release()
#         cam2.capture.release()
#         cv2.destroyAllWindows()
#         break


# from threading import Thread
# import cv2
# import time

# config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model = 'frozen_inference_graph.pb'
# model = cv2.dnn_DetectionModel(frozen_model, config_file)
# classlabel = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# class PersonTracker:
#     def __init__(self):
#         self.tracker = cv2.TrackerMIL.create()
#         self.object_id = 0
#         self.objects = {}  # Dictionary to store person trackers

#     def detect_and_track(self, frame):
#         classIndex, Confidence, box = model.detect(frame, confThreshold=0.5)
#         detected_persons = []

#         if len(classIndex) != 0:
#             for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(), box):
#                 if cIndex == 1:  # Check if the detected object is a person
#                     x, y, w, h = boxes
#                     detected_persons.append((x, y, w, h))

#         self.update_tracking(detected_persons, frame)

#     def update_tracking(self, detected_persons, frame):
#         for person_id, (tracker, bbox) in self.objects.items():
#             success, new_bbox = tracker.update(frame)
#             if success:
#                 self.objects[person_id] = (tracker, new_bbox)
#             else:
#                 # Remove tracker if tracking fails
#                 del self.objects[person_id]

#         for (x, y, w, h) in detected_persons:
#             person_id = None

#             for object_id, (_, bbox) in self.objects.items():
#                 if self.is_overlap(bbox, (x, y, w, h)):
#                     person_id = object_id
#                     break

#             if person_id is None:
#                 # Create a new tracker if not found
#                 self.object_id += 1
#                 person_id = self.object_id
#                 new_tracker = cv2.TrackerMIL.create()
#                 new_tracker.init(frame, (x, y, w, h))
#                 self.objects[person_id] = (new_tracker, (x, y, w, h))

#             # Draw the person and ID on the frame
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"Person {person_id}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)

#     @staticmethod
#     def is_overlap(bbox1, bbox2):
#         x1, y1, w1, h1 = bbox1
#         x2, y2, w2, h2 = bbox2

#         if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
#             return True
#         return False


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

# person_tracker = PersonTracker()

# while True:
#     try:
#         Myframe1 = cam1.getFrame()
#         Myframe2 = cam2.getFrame()

#         person_tracker.detect_and_track(Myframe1)
#         person_tracker.detect_and_track(Myframe2)

#         cv2.imshow('Camera 1', Myframe1)
#         cv2.imshow('Camera 2', Myframe2)
#     except:
#         print('Frame not Available')

#     if cv2.waitKey(1) == ord('q'):
#         cam1.capture.release()
#         cam2.capture.release()
#         cv2.destroyAllWindows()
#         break


# from threading import Thread
# import cv2
# import time

# config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model = 'frozen_inference_graph.pb'
# model = cv2.dnn_DetectionModel(frozen_model, config_file)
# classlabel = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# class PersonTracker:
#     def __init__(self):
#         self.tracker = cv2.TrackerMIL.create()
#         self.object_id = 0
#         self.objects = {}  # Dictionary to store person trackers

#     def detect_and_track(self, frame):
#         classIndex, Confidence, box = model.detect(frame, confThreshold=0.5)
#         detected_persons = []

#         if len(classIndex) != 0:
#             for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(), box):
#                 if cIndex == 1:  # Check if the detected object is a person
#                     x, y, w, h = boxes
#                     detected_persons.append((x, y, w, h))

#         self.update_tracking(detected_persons, frame)

#     def update_tracking(self, detected_persons, frame):
#         for person_id, (tracker, bbox) in self.objects.items():
#             success, new_bbox = tracker.update(frame)
#             if success:
#                 self.objects[person_id] = (tracker, new_bbox)
#             else:
#                 # Remove tracker if tracking fails
#                 del self.objects[person_id]

#         for (x, y, w, h) in detected_persons:
#             person_id = None

#             for object_id, (_, bbox) in self.objects.items():
#                 if self.is_overlap(bbox, (x, y, w, h)):
#                     person_id = object_id
#                     break

#             if person_id is None:
#                 # Create a new tracker if not found and assign a unique ID
#                 self.object_id += 1
#                 person_id = self.object_id
#                 new_tracker = cv2.TrackerMIL.create()
#                 new_tracker.init(frame, (x, y, w, h))
#                 self.objects[person_id] = (new_tracker, (x, y, w, h))

#             # Draw the person and ID on the frame
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"Person {person_id}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)

#     @staticmethod
#     def is_overlap(bbox1, bbox2):
#         x1, y1, w1, h1 = bbox1
#         x2, y2, w2, h2 = bbox2

#         if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
#             return True
#         return False


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

# person_tracker = PersonTracker()

# while True:
#     try:
#         Myframe1 = cam1.getFrame()
#         Myframe2 = cam2.getFrame()

#         person_tracker.detect_and_track(Myframe1)
#         person_tracker.detect_and_track(Myframe2)

#         cv2.imshow('Camera 1', Myframe1)
#         cv2.imshow('Camera 2', Myframe2)
#     except:
#         print('Frame not Available')

#     if cv2.waitKey(1) == ord('q'):
#         cam1.capture.release()
#         cam2.capture.release()
#         cv2.destroyAllWindows()
#         break

# from threading import Thread
# import cv2
# import time

# config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model = 'frozen_inference_graph.pb'
# model = cv2.dnn_DetectionModel(frozen_model, config_file)
# classlabel = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# class PersonTracker:
#     def __init__(self):
#         self.tracker = cv2.TrackerMIL.create()
#         self.object_ids = set()  # Set to store assigned IDs

#     def detect_and_track(self, frame):
#         classIndex, Confidence, box = model.detect(frame, confThreshold=0.5)
#         detected_persons = []

#         if len(classIndex) != 0:
#             for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(), box):
#                 if cIndex == 1:  # Check if the detected object is a person
#                     x, y, w, h = boxes
#                     detected_persons.append((x, y, w, h))

#         self.update_tracking(detected_persons, frame)

#     def update_tracking(self, detected_persons, frame):
#         for person_id in list(self.object_ids):
#             success, new_bbox = self.tracker.update(frame)
#             if success:
#                 self.tracker = cv2.TrackerMIL.create()
#                 self.tracker.init(frame, new_bbox)
#             else:
#                 # Remove tracker if tracking fails
#                 self.object_ids.remove(person_id)

#         for (x, y, w, h) in detected_persons:
#             person_id = None

#             for object_id in range(1, 1000):
#                 if object_id not in self.object_ids:
#                     person_id = object_id
#                     self.object_ids.add(person_id)
#                     break

#             if person_id is not None:
#                 # Create a new tracker and assign the ID
#                 new_tracker = cv2.TrackerMIL.create()
#                 new_tracker.init(frame, (x, y, w, h))
#                 self.tracker = new_tracker

#             # Draw the person and ID on the frame
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"Person {person_id}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)


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

# person_tracker = PersonTracker()

# while True:
#     try:
#         Myframe1 = cam1.getFrame()
#         Myframe2 = cam2.getFrame()

#         person_tracker.detect_and_track(Myframe1)
#         person_tracker.detect_and_track(Myframe2)

#         cv2.imshow('Camera 1', Myframe1)
#         cv2.imshow('Camera 2', Myframe2)
#     except:
#         print('Frame not Available')

#     if cv2.waitKey(1) == ord('q'):
#         cam1.capture.release()
#         cam2.capture.release()
#         cv2.destroyAllWindows()
#         break


# from threading import Thread
# import cv2
# import time

# config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model = 'frozen_inference_graph.pb'
# model = cv2.dnn_DetectionModel(frozen_model, config_file)
# classlabel = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# class PersonTracker:
#     def __init__(self):
#         self.tracker = cv2.TrackerMIL_create()
#         self.object_ids = {}  # Dictionary to store assigned IDs

#     def detect_and_track(self, frame, camera_id):
#         classIndex, Confidence, box = model.detect(frame, confThreshold=0.5)
#         detected_persons = []

#         if len(classIndex) != 0:
#             for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(), box):
#                 if cIndex == 1:  # Check if the detected object is a person
#                     x, y, w, h = boxes
#                     detected_persons.append((x, y, w, h))

#         if camera_id == 1:
#             self.update_tracking(detected_persons, frame)

#     def update_tracking(self, detected_persons, frame):
#         for person_id, (tracker, bbox) in list(self.object_ids.items()):
#             success, new_bbox = tracker.update(frame)
#             if success:
#                 self.object_ids[person_id] = (tracker, new_bbox)
#             else:
#                 # Remove tracker if tracking fails
#                 del self.object_ids[person_id]

#         for (x, y, w, h) in detected_persons:
#             person_id = None

#             if not self.object_ids:
#                 person_id = 1
#             else:
#                 max_id = max(self.object_ids.keys())
#                 person_id = max_id + 1

#             new_tracker = cv2.TrackerMIL_create()
#             new_tracker.init(frame, (x, y, w, h))
#             self.object_ids[person_id] = (new_tracker, (x, y, w, h))

#             # Draw the person and ID on the frame
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"Person {person_id}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)

# person_tracker = PersonTracker()

# class VStream:
#     def __init__(self, src, camera_id):
#         self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#         self.thread = Thread(target=self.update, args=(camera_id,))
#         self.thread.daemon = True
#         self.thread.start()

#     def update(self, camera_id):
#         while True:
#             _, self.frame = self.capture.read()
#             if camera_id == 1:
#                 person_tracker.detect_and_track(self.frame, camera_id)

#     def getFrame(self):
#         return self.frame

# flip = 2
# dispW = 640
# dispH = 480

# cam1 = VStream(0, 0)  # Assign camera_id 0 to camera 1
# cam2 = VStream(1, 1)  # Assign camera_id 1 to camera 2

# # person_tracker = PersonTracker()

# while True:
#     try:
#         Myframe1 = cam1.getFrame()
#         Myframe2 = cam2.getFrame()

#         cv2.imshow('Camera 1', Myframe1)
#         cv2.imshow('Camera 2', Myframe2)
#     except:
#         print('Frame not Available')

#     if cv2.waitKey(1) == ord('q'):
#         cam1.capture.release()
#         cam2.capture.release()
#         cv2.destroyAllWindows()
#         break


# from threading import Thread
# import cv2
# import torch
# from ultralytics import YOLO

# # Load YOLOv8n model
# model = YOLO('yolov8n.pt')

# class VStream:
#     def __init__(self, src):
#         self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()
#         self.persons = {}  # Dictionary to store unique IDs for persons
#         self.next_person_id = 1  # Initialize the next person ID to assign

#     def update(self):
#         while True:
#             _, self.frame = self.capture.read()

#     def getFrame(self):
#         return self.frame

# def detect_and_assign_ids(frame, persons_dict, next_id):
#     results = model.track(frame)  # Detect objects using YOLOv8n
#     for result in results.plot[0]:
#         class_id = int(result[5])
#         if class_id == 0:  # Check if the detected object is a person
#             bbox = tuple(map(int, result[:4]))
#             if class_id not in persons_dict:
#                 # Assign a new unique ID to the person in Camera 0
#                 persons_dict[class_id] = next_id
#                 next_id += 1
#             unique_id = persons_dict[class_id]
#             cv2.rectangle(frame, bbox, (255, 0, 0), 2)
#             cv2.putText(frame, f"Person {unique_id}", (bbox[0] + 10, bbox[1] + 40),
#                         cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)
#     return frame

# flip = 2
# dispW = 640
# dispH = 480

# cam1 = VStream(0)
# cam2 = VStream(1)

# while True:
#     try:
#         Myframe1 = cam1.getFrame()
#         Myframe2 = cam2.getFrame()

#         for frame, frame_name in [(Myframe1, 'Camera 1'), (Myframe2, 'Camera 2')]:
#             # Detect and assign IDs in the frame
#             frame_with_ids = detect_and_assign_ids(frame.copy(), cam1.persons, cam1.next_person_id)
            
#             cv2.imshow(frame_name, frame_with_ids)
#     except:
#         print('Frame not Available')

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

object_ids_camera0 = {}  # Dictionary to store object IDs for Camera 0
next_object_id = 1  

while True:
    try:
        Myframe1 = cam1.getFrame()
        Myframe2 = cam2.getFrame()

        for frame, frame_name in [(Myframe1, 'Camera 1'), (Myframe2, 'Camera 2')]:
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            for item in results[0].names:
                if item in object_ids_camera0:
                    unique_id = object_ids_camera0[item]
                else:
                    # Assign a new unique ID for objects in Camera 0
                    unique_id = next_object_id
                    face_locations = face_recognition.face_locations(frame)
                    
                    if len(face_locations) > 0:
                        # Take the first detected face for simplicity
                        face_encodings = face_recognition.face_encodings(frame, [face_locations[0]])
                        if len(face_encodings) > 0:
                            face_encoding = face_encodings[0]
                            object_ids_camera0[item] = (unique_id, face_encoding)
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
