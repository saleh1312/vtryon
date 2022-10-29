import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp





class poseDetector:
    def __init__(self):
        self.mp_drawing=mp.solutions.drawing_utils
        self.mp_pose=mp.solutions.pose
        self.pose=self.mp_pose.Pose(min_detection_confidence=0.5,static_image_mode=True, min_tracking_confidence=0.5)
    
    
        self.model_points=np.array([
                                (0.21, 0.58, -0.06),
                                (-0.21, 0.58, -0.06), 
                                
                                (0.14,0.11,0.08),   
                                (-0.14,0.11,0.08), 
                                
                                (0.56,0.56,0.01),   
                                (-0.56,0.56,0.01),  
                                
                                (0.76,0.54,0),   
                                (-0.76,0.54,0),   
                                
                                (0.10,-0.46,0.01),   
                                (-0.10,-0.46,0.01),  
                            ])
    
    
    
    def detect(self,img):
        shape =img.shape
        
        rgbimage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgbimage.flags.writeable = False
        
        results = self.pose.process(rgbimage)
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            lshoulder = [int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y*shape[0])]
            rshoulder = [int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y*shape[0])]
            lhip = [int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y*shape[0])]
            rhip = [int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y*shape[0])]
            lelbow = [int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y*shape[0])]
            relbow = [int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y*shape[0])]
            lwrist = [int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y*shape[0])]
            rwrist = [int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y*shape[0])]
            lknee = [int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y*shape[0])]
            rknee = [int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x*shape[1]),int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y*shape[0])]

            image_points = np.array([
                tl,tr,br,bl,    
                ], dtype="double")
            
            return image_points
        
        except:
            return None

        # Recolor back to BGR
        rgbimage.flags.writeable = True
        rgbimage = cv2.cvtColor(rgbimage, cv2.COLOR_RGB2BGR)
        























class arucoDetector:
    def __init__(self):
        self.arucoDict=aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.arucoparam=aruco.DetectorParameters_create()
        self.model_points=np.array([
                                (-1.0, 1.0, -1.0),             
                                (1.0, 1.0, -1.0),       
                                (1.0,-1.0,-1.0),   
                                (-1.0,-1.0,-1.0),     
                            ])
        
    def detect(self,img):
        gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        box,ids,rej=aruco.detectMarkers(gray,
                                        self.arucoDict
                                        ,parameters=self.arucoparam)
        
        box=np.array(box)
        
        
        if len(box.shape) !=4:
            return None
        
        tl=(box[0,0,0,0],box[0,0,0,1])
        tr=(box[0,0,1,0],box[0,0,1,1])
        br=(box[0,0,2,0],box[0,0,2,1])
        bl=(box[0,0,3,0],box[0,0,3,1])
        
        
        image_points = np.array([
            tl,tr,br,bl,    
            ], dtype="double")
        
        return image_points
    
    def draw_result(self,fr,points):
        pts = points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(fr,[pts],True,(255,0,0),3)
        
    

    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        