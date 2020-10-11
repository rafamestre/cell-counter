# -*- coding: utf-8 -*-
'''
Cell counting.

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import PIL
from PIL import Image

def detect(img):
    '''
    Do the detection.
    '''
    #create a gray scale version of the image, with as type an unsigned 8bit integer
    img_g = np.zeros( (img.shape[0], img.shape[1]), dtype=np.uint8 )
    img_g[:,:] = img[:,:,0]

    #1. Do canny (determine the right parameters) on the gray scale image
    edges = cv2.Canny(img_g,50,100,L2gradient=False)
    
    #Show the results of canny
    canny_result = np.copy(img_g)
    canny_result[edges.astype(np.bool)]=0
    cv2.imshow('img',canny_result)
    cv2.waitKey(0)

    #2. Do hough transform on the gray scale image
    circles = cv2.HoughCircles(img_g,cv2.HOUGH_GRADIENT,dp=1,minDist=20,param1=90,param2=13,minRadius=20,maxRadius=35)
    circles = circles[0,:,:]
    
    #Show hough transform result
    showCircles(img, circles)
    
    #3.a Get a feature vector (the average color) for each circle
    nbCircles = circles.shape[0]
    features = np.zeros( (nbCircles,3), dtype=np.int)
    for i in range(nbCircles):
        features[i,:] = getAverageColorInCircle( img , int(circles[i,0]), int(circles[i,1]), int(circles[i,2]) ) #TODO!
    
    #3.b Show the image with the features (just to provide some help with selecting the parameters)
    
    showCircles(img, circles, [ str(features[i,:]) for i in range(nbCircles)] )            

    #3.c Remove circles based on the features
    selectedCircles = np.zeros( (nbCircles), np.bool)
    for i in range(nbCircles):
        if (150<features[i,0]<220)and(140<features[i,1]<190)and(160<features[i,2]<210):    
            selectedCircles[i]=True
            
    circles = circles[selectedCircles]

    #Show final result
    showCircles(img, circles)

                
    return circles
        
    
def getAverageColorInCircle(img, cx, cy, radius):
    '''
    Get the average color of img inside the circle located at (cx,cy) with radius.
    '''
    maxy,maxx,channels = img.shape
    aux = 0
    C = np.zeros( (3) )
    
    for x in range(int(cx-radius),int(cx+radius)):
        for y in range(int(cy-radius),int(cy+radius)):
            if (x>0)and(y>0):
                if (x<maxx)and(y<maxy):
                    if ((cx-x)**2 + (cy-y)**2 <= radius**2):
                        C = C + img[y,x,:]
                        aux = aux + 1
                            
    if aux!=0:
        C = C/aux
    return C
    
    
    
def showCircles(img, circles, text=None):
    '''
    Show circles on an image.
    @param img:     numpy array
    @param circles: numpy array 
                    shape = (nb_circles, 3)
                    contains for each circle: center_x, center_y, radius
    @param text:    optional parameter, list of strings to be plotted in the circles
    '''
    #make a copy of img
    img = np.copy(img)
    #draw the circles
    nbCircles = circles.shape[0]
    for i in range(nbCircles):
        cv2.circle(img, (int(circles[i,0]), int(circles[i,1])), int(circles[i,2]),(255,0,0), 2, 8, 0 )
    #draw text
    if text!=None:
        for i in range(nbCircles):
            cv2.putText(img, text[i], (int(circles[i,0]), int(circles[i,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255) )
    #show the result
    cv2.imshow('img',img)
    cv2.waitKey(0)    


        
if __name__ == '__main__':
    #read an image
    #img = cv2.imread('normal.jpg')
    
    #the image is resized to fit in the screen (645x860)
    #then it's saved with another name and it's read from there
    img1 = Image.open('normal.jpg')
    basewidth = 860
    wpercent = (basewidth/float(img1.size[0]))
    hsize = int((float(img1.size[1])*float(wpercent)))
    img1 = img1.resize((basewidth,hsize),PIL.Image.ANTIALIAS)
    img1.save('resized_normal.jpg')    
    img = cv2.imread('resized_normal.jpg')
    
    #print the dimension of the image
    print(img.shape)
        
    #show the image
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    #do detection
    circles = detect(img)
    
    #print result
    print("We counted "+str(circles.shape[0])+ " cells.")
    








