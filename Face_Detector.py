import cv2

# Load some pre-trained data on face frontals from open cv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('RDJ.jpeg')

# Must convert to grey scale
grey_scaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grey_scaled_image)

# Draw Rectangles around the faces
for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

#print(face_coordinates)
cv2.imshow('Face Detector', grey_scaled_image)
cv2.waitKey()

print("Code Completed")
