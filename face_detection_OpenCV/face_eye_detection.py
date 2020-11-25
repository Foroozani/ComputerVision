"""The Viola-Jones algorithm """

# Importing the libraries
import cv2

# Loading the cascades, detecting only face and eyes
face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('dataset/haarcascade_eye.xml')

# Defining a function that will do the detections
# the image should be gray scale 
def detect(gray, frame):
    """
    

    Parameters
    ----------
    gray : TYPE
        Cascade work on black and white image
    frame : TYPE
        original image

    Returns
    -------
    frame : TYPE
        same image with deteced face and eyes

    """
    
    # detectMultiScale(image, scaleFactor, minNeighbors)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:    # cordinate of upper left corner
        cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 0), 3)
        cv2.putText(frame, "FACE", (x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(250,250,250),1)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(frame, "Eye", (ex,ey),cv2.FONT_HERSHEY_COMPLEX,0.5,(250,250,250),1)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

