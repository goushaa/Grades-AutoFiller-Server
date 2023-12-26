import cv2
import numpy as np
from imutils import contours as imcnts
from module2.paperExtraction import *
from module2.bubbleSheetCorrection import *
from module2.trainDigits import *

def segmentId(code):
    contours, _ = cv2.findContours(code, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    bounding_rectangles = [rect for rect in bounding_rectangles if rect[2] * rect[3] > 100]
    bounding_rectangles = sorted(bounding_rectangles, key=lambda x: x[0])
    digits = []
        
    for rect in bounding_rectangles:
        x, y, w, h = rect
        digit = code[y:y+h, x:x+w]
        digits.append(digit)

    return digits



def extractBubbleCode(paper):
    x, y = paper.shape[:2]

    newX = (x // 3) + 10 
    newY = (y // 2) + 40 

    segment = paper[:newX, :newY]

    return segment



def extractStudentCode(paper):
    grayImage = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresholdedBinaryImage = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 10)
    contours, _= cv2.findContours(thresholdedBinaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True) 
        targetArea = 30000
        for contour in contours:
            epsilonValue= 0.01*cv2.arcLength(contour, True)
            paperContour = cv2.approxPolyDP(contour, epsilonValue, True)
            
            if len(paperContour) == 4 and cv2.contourArea(contour) > 0.2 * targetArea:
                code = imageTransform(paper, paperContour.reshape(4, 2))

    grayCode = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
    blurredCode= cv2.GaussianBlur(grayCode, (5,5),0.5)
    thresholdedBinaryCode = cv2.adaptiveThreshold(blurredCode, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 155, 10)
    negativeCode=negativeTransformation(thresholdedBinaryCode)
    eroded_image = cv2.erode(negativeCode, np.ones((2,2), np.uint8), iterations=1)
    dilated_image = cv2.dilate(eroded_image, np.ones((3, 3), np.uint8), iterations=1)
    return dilated_image



def cropCode(code):
    kernel = np.ones((10, 25), np.uint8)
    dilated_image = cv2.dilate(code, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = code[y:y+h, x:x+w]
    return cropped_image



def getStudentBubbleCode (paper):
    grayImage = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresholdedBinaryImage = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 255, 15)
    negativeImg=negativeTransformation(thresholdedBinaryImage)
    erodedImage=cv2.erode(negativeImg,np.ones((7,7)),iterations=1)
    allContours,_=cv2.findContours(negativeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    circlesContours=[]
    areasOfContours=[]
    for contour in allContours:
        (x,y,w,h)=cv2.boundingRect(contour)
        aspect_ratio=w/h
        epsilonValue= 0.01*cv2.arcLength(contour, True)
        circleContour = cv2.approxPolyDP(contour, epsilonValue, True)
        if(aspect_ratio>=0.5 and aspect_ratio<=1.5 and len(circleContour)>=4 and cv2.contourArea(contour)>30 and cv2.contourArea(contour)>1.5*cv2.arcLength(contour, True)):
            circlesContours.append(contour)
            areasOfContours.append(cv2.contourArea(contour))


    #Get Only Bubbles
    medianCircleArea=np.median(areasOfContours)
    circlesContoursTemp=[]
    for i,area  in enumerate(areasOfContours):
        if(abs(area-medianCircleArea)<=medianCircleArea*0.1):
            circlesContoursTemp.append(circlesContours[i])

    circlesContours= circlesContoursTemp

    contouredPaper=paper.copy()



    sortedContours,_=imcnts.sort_contours(circlesContours,method='top-to-bottom')
    (_,__,circleWidth,___)=cv2.boundingRect(circlesContours[0])
    (xPrev,yPrev,_,_)=cv2.boundingRect(sortedContours[0])

    firstRow=[]
    firstRow.append(xPrev)
    for contour in sortedContours[1:]:
        (x,y,_,_)=cv2.boundingRect(contour)
        if(abs(y-yPrev)>3):
            break
        firstRow.append(x)

    firstRow.sort()
    questionsNumberPerRow=1
    circlesNumber=len(firstRow)
    for i in range(1,circlesNumber):
        if(firstRow[i]-firstRow[i-1]>2.5*circleWidth):
            questionsNumberPerRow+=1

    answersNumberPerQuestion=circlesNumber//questionsNumberPerRow
    numberOfQuestions= len(circlesContours)//answersNumberPerQuestion

    studentAnswers = np.zeros(numberOfQuestions, dtype=int)
    choosenContours=[None]*numberOfQuestions
    studentAnswersValidate = np.zeros(numberOfQuestions, dtype=int)
    (xPrev,yPrev,_,_)=cv2.boundingRect(sortedContours[0])
    currRow=0
    xList=[]
    xList.append([xPrev,sortedContours[0]])
    for contour in sortedContours[1:]:
        (x,y,_,_)=cv2.boundingRect(contour)

        if(abs(y-yPrev)>3):
            xList.sort(key=lambda pair: pair[0])
            questionPerRow=1
            answer=1
            questionNum=currRow
            for i in range(len(xList)):
                if((i-1>=0) & ((xList[i][0]-xList[i-1][0])>(2.5*circleWidth))):
                    questionNum=currRow+questionPerRow
                    questionPerRow+=1
                    answer=1
                if(getChoice(xList[i][1],erodedImage)==1):
                    studentAnswers[questionNum]=answer
                    choosenContours[questionNum]=xList[i][1]
                    studentAnswersValidate[questionNum]+=1
                answer+=1
            xList.clear()
            currRow+=1

        xList.append([x,contour])
        xPrev=x
        yPrev=y

    xList.sort(key=lambda pair: pair[0])
    questionPerRow=1
    answer=1
    questionNum=currRow
    for i in range(len(xList)):
        if((i-1>=0) & ((xList[i][0]-xList[i-1][0])>(2.5*circleWidth))):
            questionNum=currRow+questionPerRow
            questionPerRow+=1
            answer=1
        if(getChoice(xList[i][1],erodedImage)==1):
            studentAnswers[questionNum]=answer
            choosenContours[questionNum]=xList[i][1]
            studentAnswersValidate[questionNum]+=1
        answer+=1
    xList.clear()

    for i in range(len(studentAnswersValidate)):
        if(studentAnswersValidate[i]!=1):
            studentAnswers[i]=-1
        else:
            studentAnswers[i]-=1

    cv2.drawContours(contouredPaper,choosenContours,-1,(255,0,0), 2)
    return contouredPaper,studentAnswers



def getCodePrediction(digits):
    arr=[]
    for i, image in enumerate(digits):
        eroded_image = cv2.erode(image, np.ones((2,2), np.uint8), iterations=1)
        resized_image = cv2.resize(eroded_image, (28, 28))
        arr.append(get_predict(resized_image))
    return arr