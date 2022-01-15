import cv2
import numpy as np
img=cv2.imread("testroad2.png")

#set convert valid if it is 0, just check the input image
valid=1


for i in range(len(img)):
    for j in range(len(img[0])):
        if (img[i][j][0]==0 and img[i][j][1]==0 and img[i][j][2]==0)==False and (img[i][j][0]==255 and img[i][j][1]==255 and img[i][j][2]==255)==False :
            valid=0

if valid==1:
    print("valid data. converting start")
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][0]==0 and img[i][j][1]==0 and img[i][j][2]==0:
                img[i][j]=np.array([255,255,255])
            elif img[i][j][0]==255 and img[i][j][1]==255 and img[i][j][2]==255:
                img[i][j]=np.array([0,0,0])
            if i > len(img)-10:
                img[i][j]=np.array([0,0,0])
cv2.imwrite("converted2.png",img)
