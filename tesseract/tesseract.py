import pytesseract
import argparse

from PIL import Image

file_name = 'test1.png'
text = pytesseract.image_to_string(Image.open(file_name))

f= open("outputFiles/"+file_name+".txt","w+")
for line in text.split('\n'):
	f.write(line)
	f.write('\n')
f.close()
