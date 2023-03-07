import cv2
import numpy as np

# Read in the image in grayscale
img = cv2.imread('lego3.png', cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
#adjust the size of needed!
params.minArea = 50
params.maxArea = 5000

# Determine which openCV version were using
if cv2.__version__.startswith('2.'):
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Detect the blobs in the image
keypoints = detector.detect(img)
# print(len(keypoints))

pts = cv2.KeyPoint_convert(keypoints)
#this will return the x y coodinate of the lego piece!

# Draw detected keypoints as red circles
imgKeyPoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)