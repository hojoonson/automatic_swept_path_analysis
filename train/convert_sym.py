import cv2
import glob

right_path=glob.glob("./racingline_test/*.png")
for element in right_path:
    filename=element.split("/")[-1].split(".")[0]
    image=cv2.imread(element)
    newimage=cv2.flip(image,1)
    cv2.imwrite("./testimages/left/"+filename+"_1.png",newimage)
