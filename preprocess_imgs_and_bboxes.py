import os

def main(parent_dir, img_id, save_dir, rotate_angle=None, resize_size=None, patch_img_size=None):
    """ Rotate, resize and patch images. Also works for images without bboxes.
    
    :param parent_dir: Images in subdirectories of parent_dir are those to be preprocessed.
    :param img_id: Ending of image files that need to be preprocessed. E.g. if you want all images ending with ".png" to be preprocessed, set img_id = ".png".
    :param save_dir: Existing directory in which preprocessed images are to be saved.
    :param rotate_angle: Angle by which to rotate images. Leave out if you don't want images rotated.
    :param resize_size: Size to which to resize images. Leave out if you don't want images resized.
    :param patch_img_size: Size to which images should be patched. Leave out if you don't want images patched.
    """
    
    subdirs = [x[0] for x in os.walk(dir)] 
    for subdir in subdirs: 
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):                                                                                          
            for file_name in files: 
                if file_name.endswith(img_id):
                    src_path = os.path.join(subdir, file_name) 
                    # Apply transforms and compose name that way.
                    trsf_name = ""
                    save_path = src_path[:-4] + end_txt + img_id[-4:]
    
