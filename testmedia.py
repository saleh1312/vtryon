import cv2
import mediapipe as mp
import numpy as np

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose



with mp_pose.Pose(min_detection_confidence=0.5,static_image_mode=True, min_tracking_confidence=0.5) as pose:
    img=cv2.imread("data\\mediapipetest\\2.jpg")
    shape =img.shape
    
    rgbimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgbimage.flags.writeable = False
    
    results = pose.process(rgbimage)
    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        lshoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y*shape[0])]
        rshoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y*shape[0])]
        lhip = [int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y*shape[0])]
        rhip = [int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y*shape[0])]
        lelbow = [int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y*shape[0])]
        relbow = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y*shape[0])]
        lwrist = [int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y*shape[0])]
        rwrist = [int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y*shape[0])]
        lknee = [int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y*shape[0])]
        rknee = [int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x*shape[1]),int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y*shape[0])]

        print(lshoulder)
    except:
        pass

    # Recolor back to BGR
    rgbimage.flags.writeable = True
    rgbimage = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2BGR)
    
    # Render detections
    mp_drawing.draw_landmarks(rgbimage, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                             )    
    
    cv2.imshow('Mediapipe Feed', rgbimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


