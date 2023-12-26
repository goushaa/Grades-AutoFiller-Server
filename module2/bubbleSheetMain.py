import cv2
import numpy as np
from imutils import contours as imcnts
import os
import pandas as pd

from module2.paperExtraction import *
from module2.bubbleSheetCorrection import *
from module2.trainDigits import *
from module2.codeExtraction import *


def fillGradesInSheet(studentCode, studentGrades):
    main_relative_path = "./flask-server/module2/"

    fileName = main_relative_path + 'Output.xlsx'

    if os.path.isfile(fileName):
        existingData = pd.read_excel(fileName)

        newData = {'Code': [studentCode]}
        newData.update({f'Q{i + 1}': [answer] for i, answer in enumerate(studentGrades)})
        new_df = pd.DataFrame(newData)

        combined_df = pd.concat([existingData, new_df], ignore_index=True)

        combined_df.to_excel(fileName, index=False)
        print(f"Excel sheet '{fileName}' updated successfully.")
    else:
        data = {'Code': [studentCode]}
        data.update({f'Q{i + 1}': [answer] for i, answer in enumerate(studentGrades)})
        df = pd.DataFrame(data)

        df.to_excel(fileName, index=False)
        print(f"Excel sheet '{fileName}' created successfully.")


def readAnswersFromFile(file_path):
    with open(file_path, 'r') as file:
        answers = [int(line.strip()) for line in file]

    return answers


def process2(img,file):
    modelAnswer = [int(line.strip()) for line in file]

    main_relative_path = "./flask-server/module2/"

    img = cv2.resize(img, (800,1000))
    paper = extractPaper(img)

    # get student bubble code (Code reading method 1)
    bubbleCode=extractBubbleCode(paper)
    studentBubbleCodeImg,studentBubbleCode=getStudentBubbleCode(bubbleCode)

    # get student written code (Code reading method 2)
    code=extractStudentCode(paper)
    croppedCode =cropCode(code)
    digits=segmentId(croppedCode)
    writtenCode=getCodePrediction(digits)
    writtenCodeStr = ''.join(writtenCode)

    # get student answers
    answersRegion=extractAnswersRegion(paper)
    answersImg,answers,grades=getStudentAnswers(answersRegion,modelAnswer)


    output_folder = f'{main_relative_path}Outputs/{writtenCodeStr}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save images in the student's folder
    cv2.imwrite(os.path.join(output_folder, 'studentBubbleCode.jpg'), studentBubbleCodeImg)
    cv2.imwrite(os.path.join(output_folder, 'studentWrittenCode.jpg'), croppedCode)
    cv2.imwrite(os.path.join(output_folder, 'studentAnswers.jpg'), answersImg)

    print("Student Bubble Code: ",studentBubbleCode)
    print("Student Written Code: ",writtenCode)
    print("Student Answers: ",answers)
    print("Student Grades: ",grades,"\n")

    fillGradesInSheet(writtenCodeStr, grades)


