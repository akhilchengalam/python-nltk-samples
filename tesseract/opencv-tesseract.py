import cv2
import pytesseract
try:
    from PIL import Image
except ImportError:
    import Image

img = cv2.imread(r'test2.png')
f= open("outputFiles/"+ "opencv-1-.txt","w+")
f.write(pytesseract.image_to_string(img))
f.close()

f= open("outputFiles/"+ 'opencv-2-' + ".txt","w+")
f.write(pytesseract.image_to_string(Image.fromarray(img)))
f.close()