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





focal_lengh=900
size=512



pygame.init()
display = ( size,size)
screen=pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
glEnable(GL_TEXTURE_2D);
glEnable(GL_DEPTH_TEST)






glMatrixMode(GL_PROJECTION)

#glLoadIdentity()
fov=2*(np.arctan(0.5*display[0]/900)*180/np.pi)
gluPerspective(fov, 1, 0.1, 100.0)







back = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, back)
glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)


#glMatrixMode(GL_MODELVIEW)
#glLoadIdentity()




def cube():
    
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
    
def image():
    frame=cv2.imread("h1.jpg")
    im=cv2.flip(frame,0)
    im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    

    im=im.astype(np.float32)
    
    
    glBindTexture(GL_TEXTURE_2D, back)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, im.shape[0], im.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, im)
    
    
    z=float(100)
    x=(np.tan( (float(fov)/2.0) *(np.pi/180)) )*z
    glBegin(GL_QUADS)
    glTexCoord2f(0,1); glVertex3fv((-x, x, -z))
    glTexCoord2f(1,1); glVertex3fv((x, x, -z))
    glTexCoord2f(1,0); glVertex3fv((x, -x, -z))
    glTexCoord2f(0,0); glVertex3fv((-x, -x, -z))
    glEnd()
    
    glBindTexture(GL_TEXTURE_2D,0)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
            

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    image()
            
    pygame.display.flip()
    
    pygame.time.wait(10)
        
        
        



















