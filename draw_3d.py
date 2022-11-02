import pygame
from pygame.locals import *
import sys
import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from vision import computer_vision
from objloder import *
from PIL import Image
from PIL import ImageOps

class draw3d:
    def draw_cube(self):
        glBegin(GL_QUADS)
        
        glColor3f(0.0,1.0,0.0)
        glVertex3fv((-1.0,1.0,-1.0))
        glVertex3fv((1.0,1.0,-1.0))
        glVertex3fv((1.0,-1.0,-1.0))
        glVertex3fv((-1.0,-1.0,-1.0))
        
        glColor3f(1.0,0.0,0.0)
        glVertex3fv((-1.0,1.0,-1.0))
        glVertex3fv((1.0,1.0,-1.0))
        glVertex3fv((1.0,1.0,1.0))
        glVertex3fv((-1.0,1.0,1.0))
        
        glColor3f(0.0,0.0,1.0)
        glVertex3fv((-1.0,1.0,1.0))
        glVertex3fv((1.0,1.0,1.0))
        glVertex3fv((1.0,-1.0,1.0))
        glVertex3fv((-1.0,-1.0,1.0))
        
        
    
        glEnd()
        glColor3f(1.0,1.0,1.0) 
    def draw_obj(self):
        
        glBindTexture(GL_TEXTURE_2D,self.obj)
    
        glBegin(GL_TRIANGLES)
        for i in range(0,len(self.ver),8):
            v1=(self.ver[i],self.ver[i+1],self.ver[i+2])
            t1=(self.ver[i+3],self.ver[i+4])
            glTexCoord2f(t1[0],t1[1]); glVertex3fv(v1)
        
        glEnd()
        glBindTexture(GL_TEXTURE_2D,0)
        
    
        
    def draw_background(self,frame):

        
  
        im=cv2.flip(frame,0)
  
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        
    
        im=im.astype(np.float32)

        glBindTexture(GL_TEXTURE_2D, self.back)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, im.shape[0], im.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, im)
        
        
        x=self.x
        z=self.z

        glBegin(GL_QUADS)
        glTexCoord2f(0,1); glVertex3fv((-x, x, -z))
        glTexCoord2f(1,1); glVertex3fv((x, x, -z))
        glTexCoord2f(1,0); glVertex3fv((x, -x, -z))
        glTexCoord2f(0,0); glVertex3fv((-x, -x, -z))
        glEnd()

        
        glBindTexture(GL_TEXTURE_2D, 0)
        
    def draw(self,frame,mat):
        
        self.draw_background(frame)
        
        glLoadMatrixd(mat)
        
        self.draw_obj()
        glLoadIdentity()
         
        glFlush()
        
    def __init__(self,focal_lengh,size):

        self.size=size
        self.video = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (size,size))
        
        self.so=0
        pygame.init()
        display = ( size,size)
        self.screen=pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
        
        glEnable(GL_TEXTURE_2D);
        
        glEnable(GL_DEPTH_TEST)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        fov=2*(np.arctan(0.5*size/focal_lengh)*180/np.pi)
        gluPerspective(fov, 1, 0.1, 101.0)
        
        
        self.z=float(100)
        self.x=(np.tan( (float(fov)/2.0) *(np.pi/180)) )*self.z
        
        
        
        self.back = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.back)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        
        
        
        self.obj = glGenTextures(1)
        obj_image = cv2.imread(r'data\gta.png');
        obj_image=cv2.flip(obj_image,0)
        obj_image=cv2.cvtColor(obj_image,cv2.COLOR_BGR2RGB)
        obj_image=obj_image.astype(np.float32)
        glBindTexture(GL_TEXTURE_2D, self.obj)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, obj_image.shape[0], obj_image.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, obj_image)
    
    
        self.ind,self.ver=ObjLoader.load_model(r'data\gta.obj')
        print(self.ind)
        print(self.ver)
        
        
        glMatrixMode(GL_MODELVIEW)
        
    
    def save_video(self):
        data = glReadPixels(0, 0, self.size, self.size, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (self.size, self.size), data)
        image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
        self.video.write(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        
    def main(self,frame,mat,detect=True):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.video.release()
                pygame.quit()
                sys.exit()
                
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        if detect==True:
            self.draw(frame,mat)
        else:
            self.draw_background(frame)
            
        self.save_video()
        pygame.display.flip()
        
        pygame.time.wait(10)
        
        
opencvc=computer_vision('data\\test123.mp4',900,512)
opengl=draw3d(900,512)

while True:
    data=opencvc.read()
    
    if type(data[0])==type(None):
        opengl.video.release()
        break
    elif type(data[1])==type(None):
        
        opengl.main(data[0],None,False)
    else :
      
        opengl.main(data[0],data[1],True)
        
    
    
    
pygame.quit()
sys.exit()
    
    
    
    
    
    
    
    
    
    
    
    
    