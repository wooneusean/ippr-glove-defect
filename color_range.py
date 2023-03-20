import cv2
import numpy as np

def on_change(val):
    pass


img = cv2.imread('./img/burn_3.png')
# img = cv2.imread('./img/frosting_3.png')
img = cv2.resize(img, (500, 500))

windowName = 'image'
cv2.namedWindow(windowName)
cv2.createTrackbar('L upper', windowName, 205, 255, on_change)
cv2.createTrackbar('L lower', windowName, 200, 255, on_change)
cv2.createTrackbar('A upper', windowName, 125, 255, on_change)
cv2.createTrackbar('A lower', windowName, 115, 255, on_change)
cv2.createTrackbar('B upper', windowName, 120, 255, on_change)
cv2.createTrackbar('B lower', windowName, 115, 255, on_change)

while(1):

    l_upper = cv2.getTrackbarPos('L upper', windowName)
    l_lower = cv2.getTrackbarPos('L lower', windowName)
    a_upper = cv2.getTrackbarPos('A upper', windowName)
    a_lower = cv2.getTrackbarPos('A lower', windowName)
    b_upper = cv2.getTrackbarPos('B upper', windowName)
    b_lower = cv2.getTrackbarPos('B lower', windowName)

    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2LAB)
    upper = np.array([l_upper, a_upper, b_upper])
    lower = np.array([l_lower, a_lower, b_lower])
    img_copy = cv2.inRange(img_copy, lower, upper)
    img_copy = cv2.bitwise_and(img, img, mask=img_copy)
    print(img_copy)

    cv2.imshow(windowName, img_copy)
    cv2.waitKey(500)
