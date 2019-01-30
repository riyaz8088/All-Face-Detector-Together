## ALL FACE DETECTOR 


import numpy as np
import cv2       
import time



frame=cv2.imread('1.jpg')
##_________________________________________________________________###
## Haar Cascade Face Detector
## Give input as path to image
## Return (x,y,w,h) co-ordinates

def detect(frame):
    detector = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    out=detector.detectMultiScale(frame_gray, 1.1, 5)

    #out return [x,y,w,h] for cv2 rectangle [(x,y),(x+w,y+h)]
    cv2.rectangle(frame,(out[0][0],out[0][1]),(out[0][0]+out[0][2],out[0][1]+out[0][3]),(0,255,0),2)
    cv2.imshow('image',frame)
    cv2.waitKey(0)
    return out





def detect(frame):
    import dlib #
    detector = dlib.get_frontal_face_detector()
    frame_rgb = frame[:, :, ::-1]
    faces = detector(frame_rgb, 0)
    faces_updated = []
    for face in faces:
    (x, y, w, h) = (face.left(), face.top(), face.right()-face.left(), face.bottom()-face.top())
    faces_updated.append((x, y, w, h))

    return faces_updated

    
##DLIB MMOD DETECTOR

def detect(frame):
    import dlib # lazy loading
    detector = dlib.cnn_face_detection_model_v1( 'mmod_human_face_detector.dat')
    frame_rgb = frame[:, :, ::-1]
    faces = detector(frame_rgb, 0)
    faces_updated = []
    for face in faces:
    (x, y, w, h) = (face.rect.left(), face.rect.top(), face.rect.right()-face.rect.left(), face.rect.bottom()-face.rect.top())
    faces_updated.append((x, y, w, h))
    return faces_updated



        
## SSD RESNET FACE DETECTOR
def detect(frame):
    detector = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    faces = detector.forward()
    faces_filtered = []
    for index in range(faces.shape[2]):
        confidence = faces[0, 0, index, 2]
        if confidence > 0.5:
            box = faces[0, 0, index, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x2, y2) = box.astype("int")
            (x, y, w, h) = (x, y, x2-x, y2-y)
            cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2)
            cv2.imshow('img',frame)
            cv2.waitKey(0)

            faces_filtered.append((x, y, w, h))

    return faces_filtered





## MTCNN FACE DETECTOR  

def detect(frame):
    start_time=time.time()
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()
    faces =detector.detect_faces(frame)
    print("time required for MTCNN detection {} seconds:".format(time.time()-start_time))
    faces_updated = []
    for face in faces:
        boxd = face['box']
        (x, y, w, h) = (boxd[0], boxd[1], boxd[2], boxd[3])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('img',frame)
        cv2.waitKey(0)
        faces_updated.append((x, y, w, h))
    return faces_updated





