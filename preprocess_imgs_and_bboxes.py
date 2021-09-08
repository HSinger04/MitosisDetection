import os

def main(parent_dir, img_id, save_dir, rotate_angle=None, resize_size=None, patch_img_size=None):
    """ Rotate, resize and patch images.
    
    :param parent_dir: Images in subdirectories of parent_dir are those to be preprocessed.
    :param img_id: Ending of image files that need to be preprocessed. E.g. if you want all images ending with ".png" to be preprocessed, set img_id = ".png".
    :param save_dir: Existing directory in which preprocessed images are to be saved.
    :param rotate_angle: Angle by which to rotate images. Leave out if you don't want images rotated.
    :param resize_size: Size to which to resize images. Leave out if you don't want images resized.
    :param patch_img_size: Size to which images should be patched. Leave out if you don't want images patched.
    """
    
    
