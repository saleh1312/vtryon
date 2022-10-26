import cv2
import cv2.aruco as aruco
import numpy as np


class computer_vision:
    def __init__(self,video_name,focal_length,size):
        self.size=size
        self.video=cv2.VideoCapture(video_name)
        self.arucoDict=aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.arucoparam=aruco.DetectorParameters_create()
        
        self.model_points=np.array([
                                (-1.0, 1.0, -1.0),             
                                (1.0, 1.0, -1.0),       
                                (1.0,-1.0,-1.0),   
                                (-1.0,-1.0,-1.0),     
                            ])
    
        center = (size/2, size/2)
        self.camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
        self.dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
    def detect_aruco(self,img):
        gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        box,ids,rej=aruco.detectMarkers(gray,
                                        self.arucoDict
                                        ,parameters=self.arucoparam)
        
        return box,ids
    
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

    def detect_matrix(self,tl,tr,br,bl):
        image_points = np.array([
            tl,tr,br,bl,    
            ], dtype="double")
        
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs)

        return self.convert_to_opengl(rotation_vector, translation_vector)
    def read(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
        
            if ret==False:
                return [None]
            
            frame=cv2.resize(frame,(self.size,self.size))
            box,ids=self.detect_aruco(frame)
            box=np.array(box)
            if len(box.shape) !=4:
                return [frame ,None]
            
            tl=(box[0,0,0,0],box[0,0,0,1])
            tr=(box[0,0,1,0],box[0,0,1,1])
            br=(box[0,0,2,0],box[0,0,2,1])
            bl=(box[0,0,3,0],box[0,0,3,1])
            
            return [frame ,self.detect_matrix(tl,tr,br,bl),tl,tr,br,bl]
        else:
            return [None]
        
    def main(self):
        while(True):
            data=self.read()
            if type(data[0])==type(None):
                break
            
            fr=data[0]


            tl,tr,br,bl =data[2],data[3],data[4],data[5],
            pts = np.array([tl, tr, 
                br, bl],
               np.int32)
    
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(fr,[pts],True,(255,0,0),3)
            cv2.imshow('frame',fr)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        self.video.release()
        cv2.destroyAllWindows()
        
if __name__ =="__main__":
        
    obj=computer_vision(r'data\video2.mp4',900,512)
    obj.main()