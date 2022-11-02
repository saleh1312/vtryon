import cv2
import cv2.aruco as aruco
import numpy as np
from detectors import arucoDetector,poseDetector

class computer_vision:
    def __init__(self,video_name,focal_length,size):
        self.size=size
        self.video=cv2.VideoCapture(video_name)
        
        self.detector=poseDetector()
        
       
        center = (size/2, size/2)
        self.camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
    
    
    def convert_to_opengl( self,rvecs,tvecs):

    # build view matrix
        rmtx = cv2.Rodrigues(rvecs)[0]
   
 
        view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],tvecs[0][0]],
                            [rmtx[1][0],rmtx[1][1],rmtx[1][2],tvecs[1][0]],
                            [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvecs[2][0]],
                            [0.0       ,0.0       ,0.0       ,1.0    ]])
 
        inverse_matrix = np.array([[ 1.0, 1.0, 1.0, 1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [-1.0,-1.0,-1.0,-1.0],
                               [ 1.0, 1.0, 1.0, 1.0]])
 
        view_matrix = view_matrix * inverse_matrix
 
        view_matrix = np.transpose(view_matrix)
 
        return view_matrix

    def detect_matrix(self,image_points):
    
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.detector.model_points, image_points, self.camera_matrix, self.dist_coeffs)

        return self.convert_to_opengl(rotation_vector, translation_vector)
    
    def read(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
        
            if ret==False:
                return [None]
            
            frame=cv2.resize(frame,(self.size,self.size))
            image_points=self.detector.detect(frame)
            
            if type(image_points)==type(None):
                return [frame,None]
            
            return [frame ,self.detect_matrix(image_points),image_points]
        else:
            return [None]
        
    def main(self):
        while(True):
            data=self.read()
            if type(data[0])==type(None): # if video ends
                break
            elif type(data[1])==type(None):  # if video not ends but no points detected
                cv2.imshow('frame',data[0])
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue

            fr=data[0]


            self.detector.draw_result(fr, data[2])
            cv2.imshow('frame',fr)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        self.video.release()
        cv2.destroyAllWindows()
        
if __name__ =="__main__":
        
    obj=computer_vision(r'data\mediapipetest\vid.wmv',900,512)
    obj.main()