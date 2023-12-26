import cv2
import numpy as np
from imutils import contours as imcnts
from module2.paperExtraction import *

def negativeTransformation(image):
    negativeImage =255-image       
    return negativeImage


def getChoice(contour, erodedImage):
    x, y, w, h = cv2.boundingRect(contour)
    choiceRegion = erodedImage[y:y+h, x:x+w]
    sum=np.sum(choiceRegion == 255)
    return sum>=10


def extractAnswersRegion(paper):
    x, y = paper.shape[:2]
    newX = (x // 3)
    segment = paper[newX:, :]
    return segment



def getStudentAnswers (paper,modelAnswer):
    grayImage = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    thresholdedBinaryImage = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 83, 12)
    negativeImg=negativeTransformation(thresholdedBinaryImage)
    erodedImage=cv2.erode(negativeImg,np.ones((6,6)),iterations=1)
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
    cv2.drawContours(contouredPaper,circlesContours,-1,(0,0,255), 2)


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
    studentAnswersContours = [None] * numberOfQuestions
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
                    questionNum=currRow+15*questionPerRow
                    questionPerRow+=1
                    answer=1
                if(getChoice(xList[i][1],erodedImage)==1):
                    studentAnswers[questionNum]=answer
                    studentAnswersContours[questionNum]=xList[i][1]
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
            questionNum=currRow+15*questionPerRow
            questionPerRow+=1
            answer=1
        if(getChoice(xList[i][1],erodedImage)==1):
            studentAnswers[questionNum]=answer
            studentAnswersContours[questionNum]=xList[i][1]
            studentAnswersValidate[questionNum]+=1
        answer+=1
    xList.clear()

    outputPaper=paper.copy()
    grades = np.zeros(numberOfQuestions, dtype=int)
    for i in range(len(studentAnswersValidate)):
        if(studentAnswersValidate[i]!=1):
            studentAnswers[i]=0
            grades[i]=0
        if(studentAnswers[i]==modelAnswer[i]):
            cv2.drawContours(outputPaper,studentAnswersContours[i],-1,(0,255,0), 2)
            grades[i]=1
        elif(studentAnswers[i]!=0):
            cv2.drawContours(outputPaper,studentAnswersContours[i],-1,(0,0,255), 2)
            grades[i]=0

    return outputPaper,studentAnswers,grades