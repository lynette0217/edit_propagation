import cv2
import os

image = cv2.imread("./figs/1.png")
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
print(lab)
out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
cv2.imwrite("./figs/lab.png", out)
