# import threading
# import cv2
# from deepface import DeepFace


# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# counter=0

# face_match=False

# refernceImg=cv2.imread("img.jpeg")

# def check_face(frame):
#     global face_match
#     try:
#         if DeepFace.verify(frame, refernceImg.copy())['verified']:
#             face_match=True
#         else:
#             face_match=False
#     except ValueError:
#         face_match=False

# while True:
#     ret, Frame= cap.read()

#     if ret:
#         if counter % 30==0:
#             try:
#                 threading.Thread(target=check_face, args=(Frame.copy(),)).start()
#             except ValueError:
#                 pass
#         counter+=1

#         if face_match:
#             cv2.putText(Frame, "Match!",(20,450),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
#         else:
#             cv2.putText(Frame, "No Match!",(20,450),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),3)
        
#         cv2.imshow("video",Frame)
    
#     key=cv2.waitKey(1)
#     if key==ord("q"):
#         break

# cv2.destroyAllWindows()

# import threading
# import cv2
# from deepface import DeepFace

# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# counter = 0
# face_match = False
# reference_img = cv2.imread("img.jpeg")

# def check_face(frame):
#     global face_match
#     try:
#         result = DeepFace.verify(frame, reference_img.copy())
#         if result['verified']:
#             face_match = True
#         else:
#             face_match = False
#     except Exception as e:
#         print("Error:", str(e))
#         face_match = False

# while True:
#     ret, frame = cap.read()

#     if ret:
#         if counter % 30 == 0:
#             try:
#                 threading.Thread(target=check_face, args=(frame.copy(),)).start()
#             except Exception as e:
#                 print("Error:", str(e))

#         counter += 1

#         if face_match:
#             cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
#         else:
#             cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

#         cv2.imshow("video", frame)

#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break

# # Release the video capture when done
# cap.release()
# cv2.destroyAllWindows()


# import cv2

# # Load the pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# cap0=cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1040)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while True:
#     ret, frame = cap.read()
#     ret2, frame2=cap0.read()
#     if ret:
#         # Convert the frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the grayscale frame
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
#         faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

#         if len(faces) > 0:
#             cv2.putText(frame, "Person Detected", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No Person Detected", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#         if len(faces2) > 0:
#             cv2.putText(frame2, "Person Detected", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame2, "No Person Detected", (20, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#         # Draw rectangles around detected faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         for (x, y, w, h) in faces2:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         cv2.imshow("Video", frame,frame2)
       

#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import time
from deepface import DeepFace
from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")
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

padding=20


while True:
    start_time = time.time() 
    success, img = cap.read()
    results = model(img, stream=True)
    frame,bboxs=faceBox(faceNet,img)

    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for bbox in bboxs:
            # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender=genderList[genderPred[0].argmax()]


            ageNet.setInput(blob)
            agePred=ageNet.forward()
            age=ageList[agePred[0].argmax()]


            label="{},{}".format(gender,age)
            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA) 


    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            
            face_region = img[y1:y2, x1:x2]
            
            # emotion = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False)
            # emotion_result = emotion[0]

            # Determine the dominant emotion
            
            # gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in the grayscale image
            # faces = face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # for (fx, fy, fw, fh) in faces:
            #     # Extract the face ROI
            #     face_roi = gray_face[fy: fy + fh, fx: fx + fw]
            #     face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            #     # Resize the face ROI to the input size expected by the gender model
            #     blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

            #     # Pass the face ROI through the gender model
            #     gender_model.setInput(blob)
            #     gender_preds = gender_model.forward()

            #     # confidence
            #     confidence = math.ceil((box.conf[0]*100))/100
            #     print("Confidence --->",confidence)

            #     # class name
            #     cls = int(box.cls[0])
            #     print("Class name -->", classNames[cls])

            #     # Get the gender label (0: male, 1: female)
            #     gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
                
                
            #     org = [x1, y1]
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     fontScale = 1
            #     color = (255, 0, 0)
            #     thickness = 2

            #     # Draw bounding box for YOLO object detection
            #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #     cv2.putText(img,classNames[cls], org, font, fontScale, color, thickness)
            #     cv2.putText(img,gender,(x1, y2 - 300) , font, fontScale, color, thickness)

         


            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            # print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2


           
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            cv2.putText(img,classNames[cls], org, font, fontScale, color, thickness)

            
           

            # print(emotion_result['dominant_emotion'])
            # cv2.putText(img,gender, classNames[cls], org, font, fontScale, color, thickness)
            # cv2.putText(img, str(emotion_result['dominant_emotion']),(x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    end_time = time.time()
    detection_time = end_time - start_time
    print(detection_time)
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()