import cv2
from cv2 import FONT_HERSHEY_DUPLEX
# import face_recognition
from simple_facerec import SimpleFacerec

cap = cv2.VideoCapture("test_vid/test_vid_10.mp4")

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)



    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()