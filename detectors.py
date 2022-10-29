import cv2
import cv2.aruco as aruco
import numpy as np



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
        
    
    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        