import numpy as np
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image


# Images without noise
without_noise = './assets/sample_without_noise.jpg'

img1 = np.array(Image.open(without_noise))
text = pytesseract.image_to_string(img1)

# print(text)


# Images with noise
with_noise = './assets/sample_with_noise.jpg'
img2 = np.array(Image.open(with_noise))

norm_img = np.zeros((img2.shape[0], img2.shape[1]))

img = cv2.normalize(img2, norm_img, 0, 255, cv2.NORM_MINMAX)
img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
img = cv2.GaussianBlur(img, (1, 1), 0)

# cv2.imshow('img', img)
# cv2.waitKey(0)

# text2 = pytesseract.image_to_string(img)
# print(text2)

filename = './assets/sample_without_noise.jpg'
image = cv2.imread(filename)

results = pytesseract.image_to_data(image, output_type=Output.DICT)
# print(results)


for i in range(0, len(results['text'])):
    x = results['left'][i]
    y = results['top'][i]

    w = results['width'][i]
    h = results['height'][i]

    text = results['text'][i]
    conf = int(results['conf'][i])

    if conf > 70:
        text = ''.join([c if ord(c) < 128 else '' for c in text]).strip()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

cv2.imshow('IMAGE: ', image)
cv2.waitKey(0)


