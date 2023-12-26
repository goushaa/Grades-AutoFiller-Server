import cv2
import numpy as np
import os
from imutils import contours
from module1.table_extractor import convert_image_to_grayscale, store_process_image, invert_image

def extract_cells(base,image):
    image = decrease_brightness(image, value=30)

    grey = convert_image_to_grayscale(image)
    store_process_image(base, "1_grayscaled.jpg", grey)

    thresholded_image = line_threshold_image(grey)
    store_process_image(base, "2_thresholded.jpg", thresholded_image)

    inverted_image = invert_image(thresholded_image)
    store_process_image(base, "3_inverted.jpg", inverted_image)

    inverted_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, None, iterations=1) 
    store_process_image(base, "4_closed.jpg", inverted_image)

    vertical_lines_eroded_image = erode_vertical_lines(inverted_image)
    store_process_image(base, "5_erode_vertical_lines.jpg", vertical_lines_eroded_image)

    horizontal_lines_eroded_image = erode_horizontal_lines(inverted_image)
    store_process_image(base, "6_erode_horizontal_lines.jpg", horizontal_lines_eroded_image)

    combined_image = combine_eroded_images(vertical_lines_eroded_image, horizontal_lines_eroded_image)
    store_process_image(base, "7_combined_eroded_images.jpg", combined_image)

    image_with_horizontal_lines,binary=draw_horizontal_lines(vertical_lines_eroded_image,image)
    store_process_image(base, "8_image_with_Hlines.jpg", image_with_horizontal_lines)

    image_with_all_lines,binary=draw_vertical_lines(horizontal_lines_eroded_image,image_with_horizontal_lines,binary)
    store_process_image(base, "9_image_with_all_lines.jpg", image_with_all_lines)

    cells=crop_cells(grey,binary)
    id_cells, arabic_name_cells, english_name_cells, shape_cells = store_cells(base,cells)
    return cells, id_cells, arabic_name_cells, english_name_cells, shape_cells

def store_cells(base,cells):
    id_cells = []
    arabic_name_cells = []
    english_name_cells = []
    shape_cells = []

    cnt=0
    for cell in cells:
        if cnt % 6 == 5:
            id_cells.append(cell)
        elif cnt % 6 == 4:
            arabic_name_cells.append(cell)
        elif cnt % 6 == 3:
            english_name_cells.append(cell)
        elif cnt % 6 == 2:
            shape_cells_rows = []
            shape_cells_rows.append(cells[cnt])
            shape_cells_rows.append(cells[cnt-1])
            shape_cells_rows.append(cells[cnt-2])
            shape_cells.append(shape_cells_rows)  
        store_process_image(base, "Row"+str(cnt)+".jpg", cell)
        cnt=cnt+1

    os.makedirs(base + "ids", exist_ok=True)
    cnt=0
    for cell in id_cells:
        store_process_image(base + "ids/", str(cnt+1)+".jpg", cell)
        cnt=cnt+1
   
    os.makedirs(base + "arabic_names", exist_ok=True)
    cnt=0
    for cell in arabic_name_cells:
        store_process_image(base + "arabic_names/", str(cnt+1)+".jpg", cell)
        cnt=cnt+1

    os.makedirs(base + "english_names", exist_ok=True)
    cnt=0
    for cell in english_name_cells:
        store_process_image(base + "english_names/", str(cnt+1)+".jpg", cell)
        cnt=cnt+1    
    
    os.makedirs(base + "shapes", exist_ok=True)
    cnt=0
    for i in range(len(shape_cells)):
        for j in range(3):
            store_process_image(base + "shapes/", "row"+str(i+1)+"column"+str(j+1)+".jpg", shape_cells[i][j])
            
    return id_cells, arabic_name_cells, english_name_cells, shape_cells     
    
def crop_cells(image,binary):
    #extract cells
    img=image.copy()
    binary = np.float32(binary)
    binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    binary[binary > 0.5] = 1
    binary[binary <= 0.5] = 0
    binary = np.uint8(binary)
    cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
    cells = []
    for cell_points in cnts:
        
        if (len(cell_points) == 4): 
            p1, p2, p3, p4 = cell_points
            p1 = p1[0]
            p2 = p2[0]
            p3 = p3[0]
            p4 = p4[0]
            h, w = img.shape
            # min cell width || min cell height 
            if (p2[1]-p1[1] < 0.0265522*w or p3[0]-p2[0] < 0.05534*h):
                continue
            cell = img[p1[1]:p2[1], p1[0]:p4[0]]
            kernel = np.array([[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]])
            cell = cv2.filter2D(cell, -1, kernel)
            cells.append(cell)
    return cells

def draw_horizontal_lines(vertical_lines_eroded_image,image):
    lines = cv2.HoughLines(vertical_lines_eroded_image, 1, np.pi/180, 300)
    image_with_horizontal_lines=image.copy()
    binary = np.ones(image.shape)
    # Draw the detected lines on the original image
    linesH = []
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if theta != 0: # not veritcal 
                    linesH.append((rho, theta))
    linesH.sort()

    if linesH is not None:
        for rho, theta in linesH:
            
            if (theta > 1.58 or theta < 1.57):
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))  
            y1 = int(y0 + 3000 * (a))   
            x2 = int(x0 - 3000 * (-b))  
            y2 = int(y0 - 3000 * (a))   
            cv2.line(image_with_horizontal_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw the line on the original image
            cv2.line(binary, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return image_with_horizontal_lines,binary

def draw_vertical_lines(horizontal_lines_eroded_image,image_with_horizontal_lines,binary):
    image_with_all_lines = horizontal_lines_eroded_image.copy()
    lines = cv2.HoughLines(horizontal_lines_eroded_image, 1, np.pi/180, 300)
    linesV = []
    for line in lines:
        for rho, theta in line:
            if theta == 0:
                linesV.append((rho, theta))
    linesV.sort()
    for rho, theta in linesV:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 3000*(-b))
        y1 = int(y0 + 3000*(a))
        x2 = int(x0 - 3000*(-b))
        y2 = int(y0 - 3000*(a))
        cv2.line(image_with_horizontal_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(binary, (x1, y1), (x2, y2), (0, 0, 0), 3)
    return image_with_horizontal_lines,binary
    
def line_threshold_image(grey):
    thresholded_image = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)[1]
    return thresholded_image

def erode_vertical_lines(inverted_image):
    hor = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=10)
    return vertical_lines_eroded_image

def erode_horizontal_lines(inverted_image):
    ver = np.array([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])
    horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=10)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=10)
    return horizontal_lines_eroded_image

def combine_eroded_images(vertical_lines_eroded_image, horizontal_lines_eroded_image):
    combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
    return combined_image

def decrease_brightness(image, value=30):
    img_float = image.astype(np.float32)
    brightened_image = img_float - value
    brightened_image = np.clip(brightened_image, 0, 255)
    brightened_image = brightened_image.astype(np.uint8)
    return brightened_image