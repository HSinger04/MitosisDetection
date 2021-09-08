import os
from csv import reader, writer
from PIL import Image
import torch

def bmp_to_png(dir, end_txt):
    subdirs = [x[0] for x in os.walk(dir)] 
    for subdir in subdirs: 
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):                                                                                          
            for file_name in files: 
                # TODO: marker for bbox csv files. e.g. bbox
                if file_name.endswith(".bmp") and not file_name.endswith(end_txt + ".png"):
                    src_path = os.path.join(subdir, file_name) 
                    save_path = src_path[:-4] + end_txt +  ".png"
                    bmep_img = Image.open(src_path)
                    bmep_img.save(save_path, "png")
                    os.remove(src_path)
                    
                    

def get_bboxes(src_path, to_height_and_width=True):
    """
    :param src_path: Path to bbox csv file
    :param height_and_width: True iff src_path's file contains right_x and bottom_y and they need to be converted to width and height.
    """

    bboxes = []

    with open(src_path, "r") as f:
        csv_lines = reader(f)

        for csv_line in csv_lines:
            bbox = []
            csv_line = torch.torch(csv_line).astype(torch.int)
            left_x = csv_line[::2].min()
            right_x_or_width = csv_line[::2].max()
            top_y = csv_line[1::2].min()
            bottom_y_or_height = csv_line[1::2].max()
            if to_height_and_width:
                right_x_or_width = pts2length(top_y, bottom_y_or_height)
                bottom_y_or_height = pts2length(left_x, right_x_or_width)

            # store left_x, top_y, width, height
            bbox.append(left_x) 
            bbox.append(top_y)
            bbox.append(right_x_or_width)
            bbox.append(bottom_y_or_height)

            bboxes.append(bbox)

        return bboxes

def save_bboxes(src_path, save_path):

    if os.path.isfile(save_path):
        raise ValueError("save_path already exists.") 

    bboxes = get_bboxes(src_path)    

    with open(save_path, "w") as f:
        csv_writer = writer(f)
        for bbox in bboxes:
            csv_writer.writerow(bbox)

def create_bboxes(dir, end_txt):
    subdirs = [x[0] for x in os.walk(dir)] 
    for subdir in subdirs: 
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):                                                                                          
            for file_name in files: 
                # TODO: marker for bbox csv files. e.g. bbox
                if file_name.endswith(".csv") and not file_name.endswith(end_txt + ".csv"):
                    src_path = os.path.join(subdir, file_name) 
                    save_path = src_path[:-4] + "_" + end_txt +  ".csv"
                    save_bbox(src_path, save_path)
