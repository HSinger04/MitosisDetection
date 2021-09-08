import torch
import torchvision
import torchvision.transforms.functional as TF
from src.utils.length_and_point import length2endpt, pts2length

def transform_bboxes(bboxes, orig_img_size, transform, transform_kwargs):
    """
    :param bboxes: bboxes whose elements are torch.tensor with left_x, top_y, width and height.
    :param orig_img_size: the size of the corresponding img before the transformation
    :param transform: what transformation from torchvision.transforms to apply
    :param new_img_size: necessary kwargs for the transformation 
    :return: transformed bbox
    """

    new_bboxes = []

    for bbox in bboxes:

        new_bbox = torch.empty(4)

        right_x = length2endpt(bbox[0], bbox[2])
        bottom_y = length2endpt(bbox[1], bbox[3])

        # create a matrix of size orig_img_size that has 1 at the 
        # top left and bottom right bbox coordinate and 0 otherwise
        orig_bbox = torch.zeros(torch.Size([3]) + orig_img_size)
        orig_bbox[0, bbox[1], bbox[0]] = 1
        orig_bbox[0, bottom_y, right_x] = 1

        transformed_bbox = transform(orig_bbox, **transform_kwargs)
        (_, new_top, new_left), (_, new_bottom, new_right) = torch.nonzero(transformed_bbox)

        new_bbox[0] = new_left
        new_bbox[1] = new_top
        new_bbox[2] = pts2length(new_left, new_right) 
        new_bbox[3] = pts2length(new_top, new_bottom) 

        new_bboxes.append(new_bbox)

    return new_bboxes


def transform_img_and_bboxes(img, bboxes, transform, transform_kwargs):
    orig_img_size = img.shape[-2:]
    new_bboxes = transform_bboxes(bboxes, orig_img_size, transform, transform_kwargs)
    new_img = transform(img, **transform_kwargs)

    return new_img, new_bboxes

def resize_img_and_bboxes(img, bboxes, new_img_size):
    return transform_img_and_bboxes(img, bboxes, TF.resize, {"size": new_img_size})

def rotate_bboxes(bboxes, orig_img_size, angle):
    """
    :param bboxes: bboxes whose elements are torch.tensor with left_x, top_y, width and height.
    :param orig_img_size: the size of the corresponding img. Must be square
    :param transform: what transformation from torchvision.transforms to apply
    :param angle: angle to rotate by counter-clockwise. Only 90, 180 and 270 are supported
    :return: rotated bboxes
    """

    if not angle in {0, 90, 180, 270}:
        raise ValueError("Angle must be 90, 180 or 270")

    if not orig_img_size[0] == orig_img_size[1]:
        raise ValueError("Image is not square")

    if angle == 0:
        return bboxes
    
    new_bboxes = []

    for new_left_x, new_top_y, new_width, new_height in bboxes:
        
        if angle == 90 or angle == 270:
            # swap top_y and left_x
            temp1 = new_left_x
            new_left_x = new_top_y
            new_top_y = temp1

            temp2 = new_width
            new_width = new_height
            new_height = temp2
        
        if angle == 180 or angle == 270:
            new_top_y = orig_img_size[0] - length2endpt(new_top_y, new_height)
            new_left_x = orig_img_size[1] - length2endpt(new_left_x, new_width)

        new_bboxes.append(torch.tensor([new_left_x, new_top_y, new_width, new_height]))    

    return new_bboxes
    

def rotate_img_and_bboxes(img, bboxes, angle):
    orig_img_size = img.shape[-2:]
    new_bboxes = rotate_bboxes(bboxes, orig_img_size, angle)
    new_img = TF.rotate(img, angle)

    return new_img, new_bboxes

# Buggy but more elegant version for rotate_img_and_bboxes. Problematic when angle = 270 or angle = 90 is used.
#def rotate_img_and_bboxes(img, bboxes, angle):
#    return transform_img_and_bboxes(img, bboxes, TF.rotate, {"angle": angle})

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
