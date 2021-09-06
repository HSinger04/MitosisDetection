import torch
import torchvision.transforms.functional as TF
from src.utils.length_and_point import length2endpt, pts2length

def img2patches(rgb_img, patch_size, angle=270):
    """
    :param rgb_img: the image to split. Must be square
    :param patch_size: individual patches will be of size patch_size x patch_size
    :param angle: the angle by which to rotate the patch image.
                  Defaults to 270 for restoring original rotation
    :return: the image patches, generated from right to left, top to botom
    """
    if not rgb_img.shape[-1] % patch_size == 0:
        raise ValueError("patch_size must cleanly divide height and width")
    if not rgb_img.shape[-1] == rgb_img.shape[-2]:
        raise ValueError("rgb_img not square")
    
    patches = rgb_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # we get num_patches_root^2 many patches
    num_patches_root = rgb_img.shape[-1] // patch_size
    patches = patches.reshape((3, num_patches_root, num_patches_root, patch_size, patch_size))
    # swap channel dim with row dim
    patches = patches.transpose(0, 1)
    # swap channel dim with col dim
    patches = patches.transpose(1, 2)
    # Make patches go from left to right, top to bottom insteaf of right to left, top to bottom
    patches = torch.flip(patches, [1])
    patches = patches.reshape(-1, 3, patch_size, patch_size)
    patches = TF.rotate(patches, angle)
    return patches


def img_and_bboxes2patches(rgb_img, bboxes, patch_size, angle=270):  
    """
    :param rgb_img: the image to split. Must be square
    :param bboxes: the bounding boxes of the image. Each entry is a quadruple
                   of left-most x, top y, width and height.
    :param patch_size: individual patches will be of size patch_size x patch_size
    :param angle: the angle by which to rotate the patch image.
                  Defaults to 270 for restoring original rotation
    :return: the image patches and their bboxes, generated from right to left, top to bottom
    """  

    if not rgb_img.shape[-1] % patch_size == 0:
        raise ValueError("patch_size must cleanly divide height and width")

    img_patches = img2patches(rgb_img, patch_size, angle=angle) 

    # we get num_patches_root^2 many patches
    num_patches_root = rgb_img.shape[-1] // patch_size
    # list for storing bboxes for each patch
    bboxes_to_patches = [[] for _ in range(num_patches_root * num_patches_root)]

    for left_x, top_y, width, height in bboxes:
        # values help in determining in which image patch the four bbox coordinate points lie.
        # -1 since consider e.g. 1x1 bbox. That bbox would have width == 1, but left_x == right_x
        right_x = length2endpt(left_x, width)
        bottom_y = length2endpt(top_y, height)

        # says in which patch the respective coordinate lies.
        patch_left_x = left_x // patch_size
        patch_right_x = right_x // patch_size
        patch_top_y = top_y // patch_size
        patch_bottom_y = bottom_y // patch_size

        # Iterate through the relevant patches from left to right, top to bottom
        for patch_x in range(patch_left_x, patch_right_x + 1):
            for patch_y in range(patch_top_y, patch_bottom_y + 1):
                # Init empty bbox current patch.  Also fill... 
                bbox_to_patch = torch.empty(4)
                # with left-most x
                if not patch_x == patch_left_x:
                    bbox_to_patch[0] = 0
                else:
                    bbox_to_patch[0] = left_x % patch_size 
                # with top y 
                if not patch_y == patch_top_y:
                    bbox_to_patch[1] = 0
                else:
                    bbox_to_patch[1] = top_y % patch_size
                # with width
                if not patch_x == patch_right_x:
                    bbox_to_patch[2] = patch_size - bbox_to_patch[0]
                else:
                    bbox_to_patch[2] = pts2length(bbox_to_patch[0], right_x % patch_size)
                # with height
                if not patch_y == patch_bottom_y:
                    bbox_to_patch[3] = patch_size - bbox_to_patch[1]
                else:
                    # +1 since consider e.g. 1x1 bbox. That bbox would have height == 1, but top_x == bottom_x
                    bbox_to_patch[3] = pts2length(bbox_to_patch[1], bottom_y % patch_size)
                
                # store the bbox
                bboxes_to_patches[patch_x + patch_y * num_patches_root].append(bbox_to_patch)

    return img_patches, bboxes_to_patches
