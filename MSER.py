import cv2
import numpy as np
from NMS import non_max_suppression_slow as nms


#Create MSER object
mser = cv2.MSER_create(_max_variation=0.8)

#Your image path i-e receipt path
img = cv2.imread('images/ticket.png')

#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis_1 = img.copy()
vis_2 = img.copy()

cv2.imshow('original img', vis_1)
cv2.waitKey(0)

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

# print(regions)
print(len(regions))     # 655 regions detected when using default paras
print(regions[0])       # every region is a polygonal
print(regions[0].shape) # first element of regions have 105 points, shape of (105,2)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions] # convexHull
print("length of hulls is ", len(hulls))

cv2.polylines(vis_1, hulls, 1, (0, 255, 0))

cv2.imshow('img with polygonal', vis_1)
cv2.waitKey(0)


rectangles = []
rectangles_points = []
for contour in hulls:
     x, y, w, h = cv2.boundingRect(contour)
     cv2.rectangle(vis_2, (x, y), (x + w, y + h), (255, 0, 0), 1)
     rec = [x, y, x+w, y+h]
     rec_points = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
     rectangles.append(np.array(rec))
     rectangles_points.append(np.array(rec_points))

cv2.imshow("rectangle", vis_2)   # just the text area are white
cv2.waitKey(0)

# use nms to filter boxes
keep = np.array(rectangles) 
pick = nms(keep, 0.5) 


# mask_1 : polylines
mask_1 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask_1, [contour], -1, (255, 255, 255), -1)

cv2.imshow("masks_1", mask_1)   # just the text area are white
cv2.waitKey(0)


# mask_2 : rectangles
mask_2 = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
for contour in rectangles_points:

    cv2.drawContours(mask_2, [contour], -1, (255, 255, 255), -1)

cv2.imshow("masks_2", mask_2)   # just the text area are white
cv2.waitKey(0)


#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask_1)

cv2.imshow("text only", text_only)
cv2.waitKey(0)


text_only = cv2.bitwise_and(img, img, mask=mask_2)

cv2.imshow("text only", text_only)
cv2.waitKey(0)
