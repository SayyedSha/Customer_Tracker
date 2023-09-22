import face_recognition
import cv2

camera1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Camera 1
camera2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)  # Camera 2

# Load a sample image with known faces (for face recognition)
# known_image = face_recognition.load_image_file("known_person.jpg")
# known_face_encoding = face_recognition.face_encodings(known_image)[0]

while True:
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if not ret1 or not ret2:
        print("Error reading frames from one or both cameras")
        break

    # Detect faces in each frame
    face_locations1 = face_recognition.face_locations(frame1)
    face_locations2 = face_recognition.face_locations(frame2)

    # You can now draw rectangles around the detected faces
    for top, right, bottom, left in face_locations1:
        cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

    for top, right, bottom, left in face_locations2:
        cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera1.release()
camera2.release()
cv2.destroyAllWindows()
