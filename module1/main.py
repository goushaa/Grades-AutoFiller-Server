import cv2
import os
import shutil
from module1.table_extractor import table_extractor, read_image
from module1.extract_cells import extract_cells
from module1.write_to_excel import write_to_excel, get_ids, get_shapes_labels, get_english_names, get_excel_labels
import numpy as np

def process1(image,alreadyMadeOCR):
    # Relative to AUTOGRADING Folder
    main_relative_path = "./flask-server/module1/"
    relative_path = main_relative_path + "process_images"

    # Recreating Process Images Folder
    shutil.rmtree(relative_path, ignore_errors=True)
    os.makedirs(relative_path, exist_ok=True)

    image = cv2.resize(image,(768,1024))

    os.makedirs(relative_path+"/paper_extractor/", exist_ok=True)
    paper_image = table_extractor(relative_path+"/paper_extractor/",image,4)

    os.makedirs(relative_path+"/table_extractor/", exist_ok=True)
    table_image = table_extractor(relative_path+"/table_extractor/", paper_image,4)

    os.makedirs(relative_path+"/cells/", exist_ok=True)
    cells, id_cells, arabic_name_cells, english_name_cells, shape_cells = extract_cells(relative_path+"/cells/",table_image)

    # Preparing Data for Excel
    id_numbers = get_ids(id_cells)
    total_labels = get_shapes_labels(shape_cells,alreadyMadeOCR)
    english_names = get_english_names(english_name_cells)
    total_labels = get_excel_labels(total_labels)

    write_to_excel(main_relative_path + "output.xlsx", id_numbers, english_names, total_labels)    

