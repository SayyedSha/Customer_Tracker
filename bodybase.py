# import cv2
# import time  


# def faceBox(faceNet,frame):
#     frameHeight=frame.shape[0]
#     frameWidth=frame.shape[1]
#     blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
#     faceNet.setInput(blob)
#     detection=faceNet.forward()
#     bboxs=[]
#     for i in range(detection.shape[2]):
#         confidence=detection[0,0,i,2]
#         if confidence>0.7:
#             x1=int(detection[0,0,i,3]*frameWidth)
#             y1=int(detection[0,0,i,4]*frameHeight)
#             x2=int(detection[0,0,i,5]*frameWidth)
#             y2=int(detection[0,0,i,6]*frameHeight)
#             bboxs.append([x1,y1,x2,y2])
#             cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
#     return frame, bboxs


# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"

# config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model='frozen_inference_graph.pb'
# file_name='label.txt'

# model=cv2.dnn_DetectionModel(frozen_model,config_file)
# faceNet=cv2.dnn.readNet(faceModel, faceProto)
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

#   
#               ]


# # print(classlabel)

# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# video=cv2.VideoCapture(0, cv2.CAP_DSHOW)
# video.set(3, 1080)
# video.set(4, 720)


# padding=20

# while True:
#     start_time = time.time() 
#     ret,img=video.read()
#     frame,bboxs=faceBox(faceNet,img)

#     classIndex, Confidence, box=model.detect(img,confThreshold=0.5)

#     if(len(classIndex)!=0):
#         for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(),box):
#             if(cIndex<=9):
#                 cv2.rectangle(img,boxes,(255,0,0),2)
#                 cv2.putText(img,classlabel[cIndex-1],(boxes[0]+10,boxes[1]+40),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),3)


    

#     for bbox in bboxs:
#         # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPred=genderNet.forward()
#         gender=genderList[genderPred[0].argmax()]


#         ageNet.setInput(blob)
#         agePred=ageNet.forward()
#         age=ageList[agePred[0].argmax()]


#         label="{},{}".format(gender,age)
#         cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
#         cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
#     end_time = time.time()
#     detection_time = end_time - start_time
#     print(detection_time)
#     cv2.imshow("Age-Gender",img)
#     k=cv2.waitKey(1)
#     if k==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()




# import cv2
# import time  
# import face_recognition

# def faceBox(faceNet,frame):
#     frameHeight=frame.shape[0]
#     frameWidth=frame.shape[1]
#     blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
#     faceNet.setInput(blob)
#     detection=faceNet.forward()
#     bboxs=[]
#     for i in range(detection.shape[2]):
#         confidence=detection[0,0,i,2]
#         if confidence>0.7:
#             x1=int(detection[0,0,i,3]*frameWidth)
#             y1=int(detection[0,0,i,4]*frameHeight)
#             x2=int(detection[0,0,i,5]*frameWidth)
#             y2=int(detection[0,0,i,6]*frameHeight)
#             bboxs.append([x1,y1,x2,y2])
#             cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
#     return frame, bboxs


# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"

# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"

# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"

# config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# frozen_model='frozen_inference_graph.pb'
# file_name='label.txt'

# model=cv2.dnn_DetectionModel(frozen_model,config_file)
# model2=cv2.dnn.readNetFromTensorflow(frozen_model,config_file)
# faceNet=cv2.dnn.readNet(faceModel, faceProto)
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList = ['Male', 'Female']

# classlabel=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               ]


# # print(classlabel)

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

# padding = 20

# while True:
#     person_counts = [0] * num_cameras 
#     for camera_index, video in enumerate(video_captures):
#         start_time = time.time()
#         ret, img = video.read()
#         # frame, bboxs = faceBox(faceNet, img)
#         recog=face_recognition.face_locations(img)
        
#         # Perform object detection using the model
#         blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
#         model2.setInput(blob)
#         detections = model2.forward()

#         # Count the number of persons detected in this camera's stream
#         for detection in detections[0, 0, :, :]:
#             class_id = int(detection[1])
#             confidence = detection[2]
#             if class_id == 1 and confidence > 0.5:  # Class 1 corresponds to "person"
#                 person_counts[camera_index] += 1

#         classIndex, Confidence, box=model.detect(img,confThreshold=0.5)

#         for top, right, bottom, left in recog:   
#             cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            


#         if(len(classIndex)!=0):
#             for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(),box):
#                 if(cIndex<=9):
#                     cv2.rectangle(img,boxes,(255,0,0),2)
#                     cv2.putText(img,classlabel[cIndex-1],(boxes[0]+10,boxes[1]+40),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),3)


        

#         # for bbox in bboxs:
#         #     # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
#         #     face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#         #     blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
#         #     genderNet.setInput(blob)
#         #     genderPred=genderNet.forward()
#         #     gender=genderList[genderPred[0].argmax()]


#         #     ageNet.setInput(blob)
#         #     agePred=ageNet.forward()
#         #     age=ageList[agePred[0].argmax()]


#         #     label="{},{}".format(gender,age)
#         #     cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
#         #     cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)

#     for camera_index, count in enumerate(person_counts):
#         print(f"Camera {camera_index} - Person Count: {count}")

#         cv2.imshow(f"Camera {camera_index}", img)

#         end_time = time.time()
#         detection_time = end_time - start_time
#         print(f"Camera {camera_index} - Detection Time: {detection_time} seconds")
#     k=cv2.waitKey(1)
#     if k==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()

import cv2
import time
import face_recognition
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash

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

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
file_name = 'label.txt'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model2 = cv2.dnn.readNetFromTensorflow(frozen_model, config_file)
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

classlabel = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]

num_cameras = 2  # Adjust this based on your setup

# Initialize cameras
video_captures = []
for camera_index in range(num_cameras):
    video = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    video.set(3, 1080)
    video.set(4, 720)
    video_captures.append(video)

padding = 20
# saved_images = []
saved_hashes = []
while True:
    person_counts = [0] * num_cameras
    for camera_index, video in enumerate(video_captures):
        start_time = time.time()
        ret, img = video.read()
        # frame, bboxs = faceBox(faceNet, img)
        recog = face_recognition.face_locations(img)

        # Perform object detection using the model
        blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)
        model2.setInput(blob)
        detections = model2.forward()

        # Count the number of persons detected in this camera's stream
        for detection in detections[0, 0, :, :]:
            class_id = int(detection[1])
            confidence = detection[2]
            if class_id == 1 and confidence > 0.5:  # Class 1 corresponds to "person"
                person_counts[camera_index] += 1
                # save_person(img, [int(detection[3] * img.shape[1]), int(detection[4] * img.shape[0]), int(detection[5] * img.shape[1]), int(detection[6] * img.shape[0])], person_counts[camera_index],saved_hashes)

        classIndex, Confidence, box = model.detect(img, confThreshold=0.5)

        for top, right, bottom, left in recog:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        if (len(classIndex) != 0):
            for cIndex, conf, boxes in zip(classIndex.flatten(), Confidence.flatten(), box):
                if (cIndex <= 9):
                    cv2.rectangle(img, boxes, (255, 0, 0), 2)
                    cv2.putText(img, classlabel[cIndex - 1], (boxes[0] + 10, boxes[1] + 40),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)

        # Crop and save detected faces if object is a person
        for top, right, bottom, left in recog:
            face = img[top:bottom, left:right]
            if classlabel[classIndex[0] - 1] == "person":
                cv2.imwrite(f'person_{time.time()}.jpg', face)

        cv2.imshow(f"Camera {camera_index}", img)

        end_time = time.time()
        detection_time = end_time - start_time
        print(f"Camera {camera_index} - Detection Time: {detection_time} seconds")
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
