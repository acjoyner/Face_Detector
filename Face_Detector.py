from random import randrange

import cv2

# Load some pre-trained data on face frontals from open cv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces link
# img = cv2.imread('RDJ.jpeg')


# capture video images for webcam (0 is the default webcam)
webcam = cv2.VideoCapture(0)

# capture video images for saved video (give the path of the video for any video)
# video = cv2.VideoCapture('Screen Recording 2020-12-18 at 9.19.43 AM.mov')

while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grey scale
    grey_scaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grey_scaled_image)

    # Draw Rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 10)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break
    print("Code Completed")
