import cv2
import numpy as np

def arrangeCorners(corners):
	rect = np.zeros((4, 2), dtype = "float32")

	sum = corners.sum(axis = 1)
	diff = np.diff(corners, axis = 1)
	
	rect[0] = corners[np.argmin(sum)]
	rect[1] = corners[np.argmin(diff)]
	rect[2] = corners[np.argmax(sum)]
	rect[3] = corners[np.argmax(diff)]

	return rect


def imageTransform(img, pts):
	rect = arrangeCorners(pts)
	(tl, tr, br, bl) = rect

	width1 = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	width2 = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	height1 = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	height2 = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	
	maxWidth = max(int(width1), int(width2))
	maxHeight = max(int(height1), int(height2))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	
	# compute the perspective transform matrix and then apply it
	persMatrix = cv2.getPerspectiveTransform(rect, dst)
	warpedImg = cv2.warpPerspective(img, persMatrix, (maxWidth, maxHeight))

	return warpedImg



def extractPaper(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresholdedBinaryImage = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _= cv2.findContours(thresholdedBinaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True) 
        targetArea = image.shape[0] * image.shape[1]
        for contour in contours:
            epsilonValue= 0.01*cv2.arcLength(contour, True)
            paperContour = cv2.approxPolyDP(contour, epsilonValue, True)
      
            if len(paperContour) == 4 and cv2.contourArea(contour) > 0.2 * targetArea:
                paper = imageTransform(image, paperContour.reshape(4, 2))
    return paper
