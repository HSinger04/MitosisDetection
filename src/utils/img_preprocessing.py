from torchvision.transforms.functional import rotate

def img2patches(rgb_img, patch_size, angle=270):
    """

    :param rgb_img: the image to split
    :param patch_size: individual patches will be of size patch_size x patch_size
    :param angle: the angle by which to rotate the patch image.
                  Defaults to 270 for restoring original rotation
    :return: the image patches, generated from right to left, top to botom
    """
    if not rgb_img.shape[-1] % patch_size == 0:
        raise ValueError("patch_size must cleanly divide height and width")
    
    patches = rgb_img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # TODO: What do the two "2"s mean in the line below? Seems like row and col dim when I look at the lines below, so prolly 
    # needs to be changed.
    patches = patches.reshape((3, 2, 2, patch_size, patch_size))
    # swap channel dim with row dim
    patches = patches.transpose(0, 1)
    # swap channel dim with col dim
    patches = patches.transpose(1, 2)
    patches = patches.reshape(-1, 3, patch_size, patch_size)
    patches = rotate(patches, angle)
    return patches
