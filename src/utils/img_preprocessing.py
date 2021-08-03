from torchvision.transforms.functional import rotate

def img2patches(rgb_img, patch_size, angle=270):
    """
    :param rgb_img: the image to split. Must be square
    :param patch_size: individual patches will be of size patch_size x patch_size
    :param angle: the angle by which to rotate the patch image.
                  Defaults to 270 for restoring original rotation
    :return: the image patches, generated from left to right, top to botom
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
    # Make patches go from left to right, top to bottom instead of right to left, top to bottom
    # TODO: Might be speed and memory-inefficient due to making a copy here!
    patches = torch.flip(patches, [1])
    patches = patches.reshape(-1, 3, patch_size, patch_size)
    patches = rotate(patches, angle)
    return patches
