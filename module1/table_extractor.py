import cv2
import numpy as np
from imutils import contours

def table_extractor(base,image,polysize):
    store_process_image(base, "1_original.jpg", image)

    grayscale_image = convert_image_to_grayscale(image)
    store_process_image(base, "2_grayscaled.jpg", grayscale_image)

    thresholded_image = threshold_image(grayscale_image)
    store_process_image(base, "3_thresholded.jpg", thresholded_image)

    inverted_image = invert_image(thresholded_image)
    store_process_image(base, "4_inverteded.jpg", inverted_image)

    dilated_image = dilate_image(inverted_image)
    store_process_image(base, "5_dialateded.jpg", dilated_image)

    contours,image_with_all_contours = find_contours(image,dilated_image)
    store_process_image(base, "6_all_contours.jpg", image_with_all_contours)

    image_with_only_rectangular_contours,rectangular_contours = filter_contours_and_leave_only_rectangles(image, contours,polysize)
    store_process_image(base, "7_only_rectangular_contours.jpg", image_with_only_rectangular_contours)

    image_with_contour_with_max_area , contour_with_max_area = find_largest_contour_by_area(image, rectangular_contours)
    store_process_image(base, "8_contour_with_max_area.jpg", image_with_contour_with_max_area)

    image_with_points_plotted, contour_with_max_area_ordered = order_points_in_the_contour_with_max_area(image,contour_with_max_area)
    store_process_image(base, "9_with_4_corner_points_plotted.jpg", image_with_points_plotted)

    new_image_width,new_image_height = calculate_new_width_and_height_of_image(image,contour_with_max_area_ordered)
    perspective_corrected_image = apply_perspective_transform(image, new_image_width, new_image_height, contour_with_max_area_ordered)
    store_process_image(base, "10_perspective_corrected.jpg", perspective_corrected_image)
    return perspective_corrected_image
    

def read_image(image_path):
    return cv2.imread(image_path)

def convert_image_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold_image(grayscale_image):
    return cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def invert_image(thresholded_image):
    return cv2.bitwise_not(thresholded_image)

def dilate_image(inverted_image):
    return cv2.dilate(inverted_image, None, iterations=1)

def find_contours(image,dilated_image):
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_all_contours = image.copy()
    cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 3)
    return contours,image_with_all_contours

def filter_contours_and_leave_only_rectangles(image, contours,polysize):
    rectangular_contours = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        height = h
        image_height =image.shape[0]
        
        if len(approx) == polysize and 0 < height < (99/100) *image_height:
            rectangular_contours.append(approx)
       
    image_with_only_rectangular_contours = image.copy()
    cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
    return image_with_only_rectangular_contours,rectangular_contours

def find_largest_contour_by_area(image, rectangular_contours):
    max_area=0
    contour_with_max_area = None
    for contour in rectangular_contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            contour_with_max_area = contour
    image_with_contour_with_max_area = image.copy()
    cv2.drawContours(image_with_contour_with_max_area, [contour_with_max_area], -1, (0, 255, 0), 3)
    return image_with_contour_with_max_area, contour_with_max_area

def order_points_in_the_contour_with_max_area(image,contour_with_max_area):
    contour_with_max_area_ordered = order_points(contour_with_max_area)
    image_with_points_plotted = image.copy()
    for point in contour_with_max_area_ordered:
        point_coordinates = (int(point[0]), int(point[1]))
        image_with_points_plotted = cv2.circle(image_with_points_plotted, point_coordinates, 10, (0, 0, 255), -1)
    return image_with_points_plotted, contour_with_max_area_ordered    

def calculate_new_width_and_height_of_image(image,contour_with_max_area_ordered):
    existing_image_width = image.shape[1]
    existing_image_width_reduced_by_10_percent = int(existing_image_width * 0.9)
    
    distance_between_top_left_and_top_right = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[1])
    distance_between_top_left_and_bottom_left = calculateDistanceBetween2Points(contour_with_max_area_ordered[0], contour_with_max_area_ordered[3])

    aspect_ratio = distance_between_top_left_and_bottom_left / distance_between_top_left_and_top_right

    new_image_width = existing_image_width_reduced_by_10_percent
    new_image_height = int(new_image_width * aspect_ratio)
    return new_image_width,new_image_height

def apply_perspective_transform(image, new_image_width, new_image_height, contour_with_max_area_ordered):
    heightThres = 498
    widthThres = 621
    pts1 = np.float32(contour_with_max_area_ordered)
    pts2 = np.float32([[0, 0], [new_image_width, 0], [new_image_width, new_image_height], [0, new_image_height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_corrected_image = cv2.warpPerspective(image, matrix, (new_image_width, new_image_height))
    height,width, c1 = perspective_corrected_image.shape
    if(height == heightThres and width == widthThres):
        perspective_corrected_image = image
    return perspective_corrected_image

def calculateDistanceBetween2Points(p1, p2):
    dis = ((p2[0] - p1[0]) * 2 + (p2[1] - p1[1]) * 2) ** 0.5
    return dis

def order_points(points):
    # Logic:
    # The sum of coordinates is minimal for the top-left corner and maximal for the bottom-right corner.
    # The difference between x-coordinates is minimal for the top-right corner and maximal for the bottom-left corner.
    points = points.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum,
    # the bottom-right point will have the largest sum
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    # top-right point will have the smallest difference,
    # the bottom-left will have the largest difference
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect

def store_process_image(base,file_name, image):
    path = base + file_name
    cv2.imwrite(path, image) 
