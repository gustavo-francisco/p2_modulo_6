from __future__ import print_function
import cv2
import argparse

scale_factor = 1.2
min_neighbors = 7
min_size = (50, 50)

video_capture = cv2.VideoCapture('./assets/arsene.mp4')

if not video_capture.isOpened():
    print("Error opening video file")
    exit(1)

width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = cv2.VideoWriter( './video_alterado.avi',cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y+h,x:x+w]

        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    cv2.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='./training/lbpcascade_frontalface.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='./training/haarcascade_eye_tree_eyeglasses.xml')
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    detectAndDisplay(frame)

    video_final = detectAndDisplay(frame)

    output_video.write(video_final)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

