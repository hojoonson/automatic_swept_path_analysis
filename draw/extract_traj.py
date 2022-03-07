import cv2
import numpy as np
import imutils

my_input="Mack_trucks_terrapro_my.png"
autocad_input="Mack_trucks_terrapro_auto_modify.png"
ground=cv2.imread("ground.png")
# load the image, convert it to grayscale, and blur it slightly

autocad_img=cv2.imread(autocad_input)
autocad_contour=cv2.inRange(autocad_img,np.array([255,255,255]),np.array([255,255,255]))
print(autocad_contour.shape)
print(np.sum(autocad_img)/255)
cv2.imwrite(autocad_input.split(".")[0]+"_edge.png",autocad_contour)

my_input_img=cv2.imread(my_input)
my_contour=cv2.inRange(my_input_img,np.array([255,255,255]),np.array([255,255,255]))
print(my_contour.shape)
print(np.sum(my_contour)/255)
cv2.imwrite(my_input.split(".")[0]+"_edge.png",my_contour)

ret,thresh = cv2.threshold(autocad_contour,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
autocad_edge = cv2.drawContours(ground,[cnt], 0, (255,0,0), 3)
cnt = contours[2]
autocad_edge = cv2.drawContours(autocad_edge,[cnt], 0, (255,0,0), 3)

ret,thresh = cv2.threshold(my_contour,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
my_edge = cv2.drawContours(autocad_edge,[cnt], 0, (0,0,255), 3)
cnt = contours[2]
my_edge = cv2.drawContours(my_edge,[cnt], 0, (0,0,255), 3)

my_edge = cv2.line(my_edge, (10, 20), (50, 20), (255, 0, 0), 3)
my_edge = cv2.line(my_edge, (10, 60), (50, 60), (0, 0, 255), 3)


import tqdm
or_area=0
and_area=0
for h in tqdm.tqdm(range(len(autocad_contour))):
    for w in range(len(autocad_contour[0])):
        if autocad_contour[h,w]==0 or my_contour[h,w]==0:
            or_area+=1
        if autocad_contour[h,w]==0 and my_contour[h,w]==0:
            and_area+=1

print(or_area,and_area,and_area/or_area)
fonts=cv2.FONT_HERSHEY_SIMPLEX
point=60,30
cv2.putText(my_edge,"AutoCAD Vehicle Tracking",point,fonts,1,(0,0,0),2)
point=60,70
cv2.putText(my_edge,"My Simulation",point,fonts,1,(0,0,0),2)
point=10,580
cv2.putText(my_edge,"IoU : "+str(round(and_area/or_area,3)),point,fonts,1,(0,0,0),2)
cv2.imwrite(my_input.split(".")[0]+"_edge_sum.png",my_edge)
