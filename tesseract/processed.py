import pytesseract
import argparse
import cv2

from PIL import Image
from matplotlib import pyplot as plt


img = cv2.imread(r'test2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Resize image
img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

#Thresholding(binarizing image)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#Remove noice
img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

f= open("outputFiles/"+ "processed-image.txt", "w+")
f.write(pytesseract.image_to_string(img))
f.close()
